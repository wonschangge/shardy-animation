<!-- sdy-coverage
transforms/export/test/convert_global_to_local/sdy_all_gather.mlir
transforms/export/test/convert_global_to_local/sdy_all_reduce.mlir
transforms/export/test/convert_global_to_local/sdy_all_slice.mlir
transforms/export/test/convert_global_to_local/sdy_all_to_all.mlir
transforms/export/test/convert_global_to_local/sdy_reduce_scatter.mlir
transforms/export/test/convert_global_to_local/sdy_collective_permute.mlir
-->

# L5-02 · lowering-sdy-collectives — 源 IR

**六个 `sdy.*` 集合通信算子 → StableHLO 的映射。**

| 文件 | 行数 | 用例数 |
|---|---|---|
| `sdy_all_gather.mlir` | 176 | 7 |
| `sdy_all_reduce.mlir` | 70 | 4 |
| `sdy_all_slice.mlir` | 77 | 4 |
| `sdy_all_to_all.mlir` | 96 | 4 |
| `sdy_reduce_scatter.mlir` | 138 | 5 |
| `sdy_collective_permute.mlir` | 104 | 6 |

**合计 661 行 / 30 用例。**

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 六条映射总表

| SDY 算子 | 降级为 | 关键属性 |
|---|---|---|
| `sdy.all_gather` | `stablehlo.all_gather` | `all_gather_dim`、`replica_groups`、`use_global_device_ids` |
| `sdy.all_reduce` | `stablehlo.all_reduce` | `replica_groups`、`use_global_device_ids` + **reduction 区域** |
| **`sdy.all_slice`** | **没有对应算子！** → `partition_id` + `dynamic_slice` | 查找表 |
| `sdy.all_to_all` | `stablehlo.all_to_all` | `concat_dimension`、`split_count`、`replica_groups` |
| `sdy.reduce_scatter` | `stablehlo.reduce_scatter` | `scatter_dimension`、`replica_groups` + **reduction 区域** |
| `sdy.collective_permute` | `stablehlo.collective_permute` | `source_target_pairs` |

**两个值得注意的点**：
1. **`sdy.all_slice` 是唯一的例外** —— StableHLO 没有对应的"切片"通信算子。
2. **`all_reduce` 与 `reduce_scatter` 需要 reduction 区域** —— 要指定"用什么算子归约"。

---

## 二、`sdy.all_gather` → `stablehlo.all_gather`

### 四个 RUN 行：两个选项的 2×2 组合

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,COMBINED,V1,COMBINED-V1
// RUN: sdy_opt %s -sdy-convert-global-to-local='per-dim-all-gather=true enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,PER-DIM,V1,PER-DIM-V1

// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,COMBINED,V3,COMBINED-V3
// RUN: sdy_opt %s -sdy-convert-global-to-local='per-dim-all-gather=true' | FileCheck %s --check-prefixes=CHECK,PER-DIM,V3,PER-DIM-V3
```

**读法**：`enable-rgv3` × `per-dim-all-gather` = **2 × 2 = 4 种组合**，
用四组 CHECK 前缀分别断言。

### 基础用例

```mlir
func.func @one_dim(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}){
```

```mlir
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}]> : tensor<8x16xf32>
```

```mlir
  // CHECK: return %[[GATHER]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}
```

```mlir
// CHECK-LABEL: func @one_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME:    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
```

```mlir
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>
```

**逐项读**：
- **局部类型**：输入 `1x16`、输出 `4x16` ——
  因为 `[{"x","y"}]` → `[{"x"}]`，第 0 维从"切 x×y（8 台）"变成"只切 x（2 台）"
  → 局部大小从 `8/(2×4)=1` 变成 `8/2=4` ✓
- **`all_gather_dim = 0`** —— 在第 0 维上聚合。
- **`channel_handle = <handle = 1, type = 1>`** —— 通信通道。
- **`replica_groups`**（两种表示，见下）。
- **`use_global_device_ids`** —— 用全局设备号。

### ★ `replica_groups` 的两种表示

**V1（`enable-rgv3=false`）**：

```mlir
  // V1-SAME{LITERAL}:   replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]>
```

**读法**：两组，每组 4 台设备。
- `[0,1,2,3]` —— `x=0` 的那 4 台（`y` 从 0 到 3）
- `[4,5,6,7]` —— `x=1` 的那 4 台

**为什么这样分组**：聚合的是 `y` 轴（`[{"y"}, {}]` 表示"把 `y` 收掉"），
所以**同一个 `x` 值下的 4 台设备**要互相 gather —— 分成 2 组，每组 4 台。

**V3（默认，`enable-rgv3=true`）**：

```mlir
  // V3-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y">]>
```

**读法**：用 **mesh 轴名 `"y"`** 表示 —— 语义完全等价，但**更简洁**，
且**与设备号解耦**（改 mesh 大小时不用改这里）。

**这正是 L5-01 提到的 `enable-rgv3` 选项的效果** ——
"Use StableHLO **ReplicaGroupV3** (mesh-axes based) for collectives"。

**为什么 V3 是默认**：mesh 轴表示更**可读**、更**稳定**（不依赖具体设备编号）。

---

## 三、★ `sdy.all_slice`：**唯一没有对应算子**的

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s
```

```mlir
func.func @one_dim_two_axes_xy(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>}) {
```

```mlir
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y"}]> : tensor<16x32xf32>
```

```mlir
  // CHECK-DAG: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG: %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]> : tensor<16xi64>
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 4] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x4xf32>
```

```mlir
  // CHECK: return %[[RESULT]] : tensor<16x4xf32>
  return %0 : tensor<16x32xf32>
}
```

**读法**（与 L5-01 的常量处理**同一个套路**）：
- 用 **`partition_id`**（不是 `replica_id`！）取设备号。
- **查找表** `[0, 0, 4, 4, 8, 8, ...]` —— 16 台设备各自的**起始列**。
- `dynamic_slice` 查表 → 再 `dynamic_slice` 从输入切出**自己那份**。

**为什么 `all_slice` 要这样做**：
- `all_slice` 的语义是"**把数据切开，每台设备留自己那片**"。
- 这在**单机内**就是一次**本地切片** —— **不需要跨设备通信**！
- 所以不需要通信算子，用 `dynamic_slice` 就够了。

**这与 `all_gather` 形成鲜明对比**：
| | 需要通信吗 | 实现 |
|---|---|---|
| `all_gather` | **需要**（要别人的数据） | `stablehlo.all_gather` |
| `all_slice` | **不需要**（只留自己的） | `partition_id` + `dynamic_slice` |

**查找表 `[0,0,4,4,8,8,...]` 的推导**（与 L5-01 同法）：
- `{"x","y"}` 切第 1 维：`x=2`、`y=4` → 共 8 份，`32/8 = 4` 列一份
- 但表里有 16 项 —— 因为网格是 `@mesh_2_4_2`（`x=2, y=4, z=2`，共 16 台）
- `z` 轴**不参与**第 1 维的切分（分片里没有 `z`），所以 `z` 变化时起始列**相同**
- 表按 `x` 最 major：`x=0` 起始 0、`x=1` 起始 8…… 而 `y` 每变一次加 4
- 最终：`0,0,4,4,8,8,...` —— 每个值**重复两次**（对应 `z` 的 2 个值）

**`partition_id` vs `replica_id`**：
- `replica_id`：**副本**编号（数据并行）
- `partition_id`：**分区**编号（模型并行）
- L5-01 的常量用 `replica_id`；这里用 `partition_id` —— 因为 `all_slice` 是**模型并行**的操作。

---

## 四、`all_reduce` 与 `reduce_scatter`：带 reduction 区域

### `sdy.all_reduce`

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,V1
```

```mlir
  %0 = sdy.all_reduce {"y":(1)2} %arg0 out_sharding=<@mesh_2_4, [{}, {"y":(2)2}]> : tensor<16x32xf32>
```

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[ARG0]])
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "y", sub_axis_info = (1)2>]>
  // CHECK-SAME: use_global_device_ids
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
  // CHECK: stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK: }) : (tensor<16x16xf32>) -> tensor<16x16xf32>
```

**读法**：
- **`replica_groups` 用子轴**：V3 里是 `sub_axis_info = (1)2` ——
  因为 SDY 算子用的是**子轴** `{"y":(1)2}`（L2-02 讲过子轴）。
- **reduction 区域**：`^bb0` 里是 `stablehlo.add` —— 指定"用加法归约"。
- **注意 `{LITERAL}`**：V1 那行带 `{LITERAL}` 修饰符，
  表示"按字面匹配，不做正则解释" —— 因为 `[[0, 2], ...]` 里的方括号会被 FileCheck 当成特殊字符。

### `sdy.reduce_scatter`

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.reduce_scatter"(%[[ARG0]])
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4, axes = [#stablehlo.axis_ref<name = "x">]>
  // CHECK-SAME: scatter_dimension = 1 : i64
  // CHECK-SAME: use_global_device_ids
  // CHECK: (%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK:   %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
  // CHECK:   stablehlo.return %1 : tensor<f32>
```

**读法**：
- `scatter_dimension = 1` —— 在第 1 维上**边归约边切分**。
- 同样带 **reduction 区域**（`add`）。
- `replica_groups` 用 `x` 轴（V3）。

**回顾 L4-01**：那里看到导出流水线把 `all_reduce` + `reshard` **融合成 `reduce_scatter`**。
现在看到它**降级后的样子** —— 完整的链路是：

```
reduce + sharding_constraint
  --L4-01 融合-->  sdy.reduce_scatter
  --L5-02 降级-->  stablehlo.reduce_scatter
```

---

## 五、`all_to_all` 与 `collective_permute`

### `sdy.all_to_all`

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local='enable-rgv3=false' | FileCheck %s --check-prefixes=CHECK,V1,PARTITION
```

```mlir
  // CHECK: %[[RESULT:.*]] = "stablehlo.all_to_all"(%[[ARG0]]) <{
  // PARTITION-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // REPLICA-NOT: channel_handle
  // CHECK-SAME: concat_dimension = 0 : i64
  // V1-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]>
  // V3-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_4_2, axes = [#stablehlo.axis_ref<name = "y">]>
  // CHECK-SAME: split_count = 4 : i64
```

**读法**：
- `concat_dimension = 0` + `split_count = 4` —— 在维度 0 上拼接、分成 4 份。
- **`channel_handle` 只在 PARTITION 模式下出现**（`PARTITION-SAME` / `REPLICA-NOT`）——
  说明 **replica 模式下不需要 channel**（同一副本内的通信 vs 跨副本的通信）。

### `sdy.collective_permute`

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,PARTITION
```

```mlir
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4, [{"y":(2)2}, {"x"}]> : tensor<4x8xf32>
```

```mlir
  // CHECK: %[[RES:.*]] = "stablehlo.collective_permute"(%[[ARG0]]) <{
  // PARTITION-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // REPLICA-NOT: channel_handle
  // CHECK-SAME{LITERAL}: source_target_pairs = dense<[[0, 0], [1, 4], [2, 2], [3, 6], [4, 1], [5, 5], [6, 3], [7, 7]]> : tensor<8x2xi64>
  // CHECK-SAME: }> : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2_4, [{"y":(2)2}, {"x"}]> : tensor<4x8xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  return %0 : tensor<4x8xf32>
}
```

**读法**：
- **`source_target_pairs`** 是 `collective_permute` 特有的 ——
  一对一的"从哪台发给哪台"映射。
- 读这个表：`[1, 4]` 表示"设备 1 发给设备 4"；`[4, 1]` 表示"设备 4 发给设备 1" ——
  **成对出现**，正是"交换"的语义。
- 局部类型从 `4x8` 变成 `2x4` —— 因为分片从 `[{"x"},{"y"}]` 变成 `[{"y":(2)2},{"x"}]`。

**回顾 L4-08**：那里讲 `collective_permute` 是"轴在两个维之间交换"。
现在看到它的**底层表示** —— 一张设备间的收发对照表。

---

## 六、六条映射的**共同点**

| 特征 | 说明 |
|---|---|
| **局部类型都变小** | 因为分片更细了（如 `8x16` → `1x16`） |
| **`replica_groups` 两种表示** | V1 设备号列表 / V3 mesh 轴（`enable-rgv3`） |
| **归约类带 reduction 区域** | `all_reduce` / `reduce_scatter` 要指定算子 |
| **`channel_handle` 视模式而定** | partition 模式有、replica 模式没有 |
| **`all_slice` 是例外** | 不需要通信 → `partition_id` + `dynamic_slice` |

**一句话总结**：
> **SDY 的集合通信算子降级为 StableHLO 的对应算子**；
> 唯一例外是 `all_slice` —— 它只是"本地切片"，用 `dynamic_slice` 就够。
>
> 这与 **L4-08** 构成完整链路：那里把 `reshard` **转成** `sdy.*` 通信，
> 这里把 `sdy.*` 通信**降级**成真正的硬件指令。
