<!-- sdy-coverage
transforms/export/test/insert_explicit_reshards/data_flow_ops.mlir
transforms/export/test/insert_explicit_reshards/manual_computation.mlir
transforms/export/test/insert_explicit_reshards/call_ops.mlir
transforms/export/test/insert_explicit_reshards/call_ops_enable_full_version_false.mlir
transforms/export/test/insert_explicit_reshards/func_inputs_outputs.mlir
transforms/export/test/insert_explicit_reshards/meshes.mlir
transforms/export/test/insert_explicit_reshards/single_device_sharding.mlir
transforms/export/test/insert_explicit_reshards/single_device_sharding_errors.mlir
transforms/export/test/insert_explicit_reshards/unreduced.mlir
-->

# L4-07 · reshard-structure — 源 IR

**L4 按算子族展开的最后一课**：结构性场景。覆盖 9 个文件（**1455 行 / 112 用例**）：

| 文件 | 行数 | 用例数 | 主题 |
|---|---|---|---|
| `data_flow_ops.mlir` | 171 | 7 | 区域算子（case / while / barrier） |
| `manual_computation.mlir` | 34 | 2 | 手动计算 |
| `call_ops.mlir` | 130 | 11 | 函数调用 |
| `call_ops_enable_full_version_false.mlir` | 124 | 11 | 同上（另一选项） |
| `func_inputs_outputs.mlir` | 109 | 14 | 函数边界与跨 mesh |
| `meshes.mlir` | 183 | 19 | **mesh 切换** |
| `single_device_sharding.mlir` | 241 | 10 | 单设备分片 |
| `single_device_sharding_errors.mlir` | 30 | 0 | 单设备分片的报错 |
| `unreduced.mlir` | 433 | 38 | 未归约轴的完整处理 |

---

## 一、`data_flow_ops`：reshard 插在**区域内部**

```mlir
// CHECK-LABEL: func @case
func.func @case(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}, %arg1: tensor<i32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
```

```mlir
  %0 = "stablehlo.case"(%arg1) ({
```

```mlir
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.abs %[[RESHARD]]
    %2 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
```

```mlir
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: stablehlo.return %[[RESHARD]] : tensor<210xf32>
    stablehlo.return %2 : tensor<210xf32>
```

**读法**（本课最有结构感的一处）：
- 函数参数是 `[{"x":(1)2}]`（**子轴**），而**分支内**的 `abs` 要求 `[{"x"}]`（**完整轴**）。
- reshard **插在分支内部**（`abs` 之前）—— 把子轴"补齐"成完整轴。
- `stablehlo.return` 前**又插一条** reshard 到 `[{"y"}]`（函数结果的要求）。

**关键**：区域算子（`case` / `while` / `barrier`）内部**可以有 reshard** ——
这与 L3-04 讲的"区域算子本身不带分片属性、分片写在 `data_flow_edge` 上"配合：
**边负责跨边界，reshard 负责区域内部**。

用例名（`case` / `while` / `optimization_barrier` 等 7 个）说明这一族覆盖各类区域算子。

---

## 二、`manual_computation`：区域**外**与区域**内**都要插

```mlir
func.func @manual_computation(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
```

```mlir
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: sdy.manual_computation(%[[RESHARD]])
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"y"}]>] manual_axes={} (%arg1: tensor<210xf32>) {
    %2 = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
    // CHECK: %[[RESHARD:.*]] = sdy.reshard %{{.*}} <@mesh, [{"y"}]> : tensor<210xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD]] : tensor<210xf32>
    sdy.return %2 : tensor<210xf32>
  } : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}
```

**读法**（两处 reshard）：
1. **区域外**：`%arg0` 是 `[{"x":(1)2}]`，而 `in_shardings` 要求 `[{"x"}]`
   → 在 `manual_computation` **之前**插 reshard。
2. **区域内**：`abs` 产出 `[{"x"}]`，而 `out_shardings` 要求 `[{"y"}]`
   → 在 `sdy.return` **之前**插 reshard。

**为什么内外都要**：
- 区域外是**全局世界**（完整形状），区域内的值要满足 `in_shardings`。
- 区域内是**局部世界**（L1-07），但 `out_shardings` 是区域对外的承诺 ——
  所以区域内也要把结果调整到承诺的分片。

### 带 `manual_axes` 的情形

```mlir
// CHECK-LABEL: func @manual_computation_with_manual_axes
func.func @manual_computation_with_manual_axes(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x", "y"}]>}) -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"x", "z"}]>}) {
```

```mlir
    // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyzt, [{"t"}]> : tensor<52xf32>
    // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"t"}]>]>} : tensor<52xf32>
    // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[ABS]] <@mesh_xyzt, [{"z"}]> : tensor<52xf32>
    // CHECK-NEXT: sdy.return %[[RESHARD2]] : tensor<52xf32>
```

**读法**：`manual_axes={"x"}` 冻结了 `x`，区域内的值在剩余轴上分片。
区域内的 reshard 在 `[{"t"}]` 与 `[{"z"}]` 之间切换 —— **不涉及被冻结的 `x`** ✓

---

## 三、`call_ops`：函数调用的两种模式

两个文件各 11 个用例：

| 文件 | RUN 选项 | 用例 |
|---|---|---|
| `call_ops.mlir` | `enable-full-version=true` | `call`、`call_empty_block`、`call_with_shardings` 等 |
| `call_ops_enable_full_version_false.mlir` | 默认（false） | 同样的用例 |

**`enable-full-version`** 是 L4-02 见过的选项 —— 控制是否启用"完整版"的 reshard 插入逻辑。

**关键**：`call` 的实参与形参、`return` 与调用结果都需要**分片对齐**。
这与 L3-05（函数级数据流边）和 L3-10（把函数结果的分片搬到调用点）配合：
- L3-05 提供**桥接机制**（`func_data_flow_edge`）
- L3-10 把分片**搬到调用点**（避免丢失）
- **本课**在两端分片**不一致**时插 reshard

---

## 四、`func_inputs_outputs`：函数边界与**跨 mesh**

14 个用例，用例名把场景写得很清楚：

| 用例 | 场景 |
|---|---|
| `funcop_result_sharding_does_not_match` | 函数结果分片不匹配 |
| `funcop_result_unsharded_but_different_meshes_between_return_and_func_result` | **return 与函数结果用不同 mesh** |
| `funcop_result_sharding_matches_but_different_meshes_between_return_and_func_result` | 分片匹配但 mesh 不同 |
| `funcop_result_sharding_does_not_match_different_meshes_between_return_and_func_result` | 两者都不匹配 |
| `..._multiple_results` | 多结果版本 |

**关键**：**mesh 不同**是比"分片不同"更根本的差异。
L2-01 讲过"分片只在同一网格内传播" —— 跨 mesh 时传播**根本不发生**，
所以必须靠 reshard 显式转换。

---

## 五、★ `meshes`：mesh 切换

19 个用例，用例名直接暴露了场景：

| 用例 | 场景 |
|---|---|
| `optimization_barrier_different_meshes` | 屏障两端用**不同 mesh** |
| `optimization_barrier_meshes_different_device_order` | mesh 相同但**设备顺序不同** |
| `negate_from_empty_sharding_to_iota_sharded` | 从空分片到 iota 分片 |
| `negate_from_empty_sharding_to_iota_unsharded` | 从空分片到 iota 无分片 |
| `binary_op_from_empty_sharding_to_iota_unsharded` | 二元算子版本 |

**`meshes_different_device_order` 尤其重要**：
两个 mesh 的**轴名与大小都相同**，但 `device_ids` 的**顺序不同** ——
这仍然是**两个不同的网格**（L2-01 讲过判据是"轴名与设备顺序"）。

**所以需要 reshard 来"跨网格搬运"** —— 这种搬运在硬件上对应真实的数据重排。

---

## 六、`single_device_sharding`：单设备分片

`single_device_sharding.mlir`（241 行 / 10 用例）+ `single_device_sharding_errors.mlir`（30 行）

用例名：

| 用例 | 场景 |
|---|---|
| `single_device_result_to_tiled_consumer` | 单设备结果 → 分片消费者 |
| `single_device_result_to_replicated_consumer` | 单设备结果 → 复制消费者 |
| `single_device_op0_to_single_device_op1` | 单设备 → 单设备 |
| `single_device_result_directly_returned` | 单设备结果直接返回 |
| `tiled_operand_to_single_device_consumer` | 分片操作数 → 单设备消费者 |

**背景**：L1-01 讲过 `@maximal_mesh = <[], device_ids=[0]>` ——
"单设备网格"。L2-01 又讲过 maximal 网格在传播中的特殊规则
（**不被替换、也不沿它传播**）。

**本课处理的是它的导出侧**：当一个算子在单设备网格上、而消费者在别的网格上时，
如何插 reshard 完成转换。

`single_device_sharding_errors.mlir` 则锁定**报错情形**（`-verify-diagnostics`）——
说明有些单设备分片的组合是**非法**的。

---

## 七、`unreduced`：38 个用例的完整展开

`unreduced.mlir`（433 行 / 38 用例）是**本课最大的一族** ——
它是 **L4-02 那 8 个用例的完整版**。

用例名（与 L4-02 高度重叠）：

| 用例 | 场景 |
|---|---|
| `all_reduce_on_func_input` | 函数输入带 unreduced |
| `unreduced_func_input_until_return` | 一直延迟到 return |
| `lhs_and_result_unreduced_rhs_replicated` | lhs 与结果未归约、rhs 复制 |
| `all_reduce_fully_delayed_until_return` | 完全延迟 |
| `all_reduce_delayed_to_call_site` | **延迟到调用点** |

**`all_reduce_delayed_to_call_site`** 是本课新增的维度 ——
归约不仅能在函数内延迟，还能**跨函数延迟到调用点**。
这与 L3-05 / 本课第三节的函数调用处理衔接。

**读法**：L4-02 是总纲（8 个代表），本课是完整展开（38 个）——
两者用**同名用例**，可以对照阅读。

---

## 八、9 个文件 / 112 个用例的族谱

| 族 | 文件 | 用例数 | 核心问题 |
|---|---|---|---|
| **区域内部** | `data_flow_ops`、`manual_computation` | 9 | reshard 插在区域**内**还是**外** |
| **函数调用** | `call_ops`、`call_ops_...false`、`func_inputs_outputs` | 36 | 实参/形参、return/结果的对齐 |
| **跨 mesh** | `meshes` | 19 | mesh 切换（含设备序不同） |
| **单设备** | `single_device_sharding`、`..._errors` | 10 | maximal 网格的导出侧 |
| **未归约** | `unreduced` | 38 | L4-02 总纲的完整展开 |

**一句话总结**：
> **前四族都是"边界"问题** —— 区域的边界、函数的边界、mesh 的边界、单设备与多设备的边界。
> **reshard 就是跨越这些边界的搬运。**
