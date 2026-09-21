<!-- sdy-coverage
transforms/export/test/optimize_collectives/all_to_all_fully_scattered.mlir
transforms/export/test/optimize_collectives/all_to_all_partially_scattered.mlir
-->

# L4-10 · optimize-collectives — 源 IR

**核心优化**：消除 `all_to_all` 链前的**冗余 `collective_permute`**。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/optimize_collectives/all_to_all_fully_scattered.mlir` | 133 | 7 |
| `transforms/export/test/optimize_collectives/all_to_all_partially_scattered.mlir` | 98 | 6 |

RUN 行：`-sdy-optimize-collectives`

网格：`sdy.mesh @mesh_2d = <["x"=2, "y"=2]>`

---

## 一、★ 核心优化：删掉 permute，改用 reshape

### 输入：permute + all_to_all

```mlir
func.func @two_axis_full_scatter(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}) -> (tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{}, {"y"}, {"x"}]>}) {
```

```mlir
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}, {}]> : tensor<16x8x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->2] %0 out_sharding=<@mesh_2d, [{"y"}, {}, {"x"}]> : tensor<16x8x8xf32>
  %2 = sdy.all_to_all [{"y"}: 0->1] %1 out_sharding=<@mesh_2d, [{}, {"y"}, {"x"}]> : tensor<16x8x8xf32>
  return %2 : tensor<16x8x8xf32>
}
```

**读法**（原始形态，三步）：
1. `collective_permute`：把第 0 维的 `{"x", "y"}` 换成 `{"y", "x"}`（**交换轴序**）
2. `all_to_all [{"x"}: 0->2]`：把 `x` 从第 0 维搬到第 2 维
3. `all_to_all [{"y"}: 0->1]`：把 `y` 从第 0 维搬到第 1 维

### 输出：permute **被消除**

```mlir
// CHECK-LABEL: func @two_axis_full_scatter
// CHECK-SAME:    %arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}, {}]>}
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->4] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {"y"}, {}, {}, {"x"}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->3] %[[A2A1]] out_sharding=<@mesh_2d, [{}, {}, {}, {"y"}, {"x"}]> : tensor<2x2x4x8x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{}, {"y"}, {"x"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
// CHECK:       return %[[RESHAPE_OUT]] : tensor<16x8x8xf32>
```

**读法**（优化后，四步）：
1. **`CHECK-NOT: sdy.collective_permute`** —— **permute 被删掉了**！
2. `reshape` 把 `16x8x8` 变成 **`2x2x4x8x8`** ——
   把复合轴 `{"x", "y"}` **拆成两个独立的维**（`2` 与 `2`，各带一个轴）。
3. 两个 `all_to_all` 分别搬 `x`（`0->4`）与 `y`（`1->3`）。
4. `reshape` 把 `2x2x4x8x8` 变回 `16x8x8`。

### ★ 为什么删 permute 是**安全**的

**关键**：`all_to_all` 的语义是"把**某个轴**从某个维搬到某个维" ——
参数写的是**轴名**（`{"x"}: 0->4`），而**轴在源维内的顺序不影响搬运结果**。

原来那条 `permute` 做的事只是"把 `{"x","y"}` 变成 `{"y","x"}`" ——
**只是顺序调整**。而后续的 `all_to_all` 按轴名搬运，顺序无关紧要。

**所以 permute 是冗余的，可以安全删除。**

**代价对比**：
- 原来：`permute`（一次真实通信！）+ 2 次 `all_to_all`
- 优化后：2 次 `all_to_all` + 2 次 `reshape`（**reshape 是本地操作，无通信**）

**省下了一次跨设备通信** —— 这就是这个优化的价值。

---

## 二、两类散开：full vs partial

### `fully_scattered`（全散开，7 个用例）

**定义**：复合轴里的**所有轴都被搬走**（搬到别的维上）。

```mlir
// Tests 2-axis full scatter on split dimension 0 where both axes {"x", "y"}
// are permuted and communicated to separate target dimensions.
```

**7 个用例**：

| 用例 | 场景 |
|---|---|
| `two_axis_full_scatter` | 两个轴搬到**不同的**目标维 |
| `two_axis_scatter_to_same_target_dim` | 两个轴搬到**同一个**目标维 |
| `three_axis_full_scatter` | 三个轴 |
| `two_axis_scatter_with_untouched_axis` | 带**未触及**的轴 |
| `non_major_split_dim` | 切分维**不是**最 major 的 |
| `sub_axis_full_scatter` | **子轴** |
| `downstream_all_to_all_on_other_dim` | 下游还有 `all_to_all`（在别的维上） |

### `partially_scattered`（部分散开，6 个用例）

**定义**：复合轴里**只有一个轴被搬走**，另一个**留在原维**。

```mlir
func.func @two_axis_permuted_one_scattered(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"x", "y"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2d, [{"y"}, {"x"}]>}) {
```

```mlir
  %0 = sdy.collective_permute %arg0 out_sharding=<@mesh_2d, [{"y", "x"}, {}]> : tensor<16x8xf32>
  %1 = sdy.all_to_all [{"x"}: 0->1] %0 out_sharding=<@mesh_2d, [{"y"}, {"x"}]> : tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}
```

```mlir
// CHECK-NOT:   sdy.collective_permute
// CHECK:       %[[RESHAPE_IN:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<16x8xf32>) -> tensor<2x2x4x8xf32>
// CHECK:       %[[A2A1:.*]] = sdy.all_to_all [{"x"}: 0->3] %[[RESHAPE_IN]] out_sharding=<@mesh_2d, [{}, {"y"}, {}, {"x"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[A2A2:.*]] = sdy.all_to_all [{"y"}: 1->0] %[[A2A1]] out_sharding=<@mesh_2d, [{"y"}, {}, {}, {"x"}]> : tensor<2x2x4x8xf32>
// CHECK:       %[[RESHAPE_OUT:.*]] = stablehlo.reshape %[[A2A2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2d, [{"y"}, {"x"}]>]>} : (tensor<2x2x4x8xf32>) -> tensor<16x8xf32>
```

**读法**（注释直接点明："only `x` is communicated, leaving permuted `y` on dim 0"）：
- 只有 `x` 需要搬到第 1 维（`0->3`）。
- `y` **留在第 0 维**（`1->0` —— 从 reshape 出来的第 1 维回到第 0 维）。

**为什么仍然要两条 `all_to_all`**：
- `reshape` 把 `{"x","y"}` 拆成两个维后，`y` 到了**新产生的第 1 维**。
- 但目标要求 `y` 在**第 0 维** → 所以还要一条 `all_to_all [{"y"}: 1->0]` 把它搬回去。
- 这条 all_to_all 是**本地**的（源维和目标维都是原来的第 0 维拆出来的）——
  但它仍然是 `all_to_all` 算子。

**6 个用例**：

| 用例 | 场景 |
|---|---|
| `two_axis_permuted_one_scattered` | 一个轴被搬，另一个留在原维 |
| `three_axis_permuted_two_scattered` | 三个轴，两个被搬 |
| `cyclic_three_axis_permuted_one_scattered` | **循环**置换 |
| `non_major_split_dim_partial_scatter` | 切分维不是最 major |
| `sub_axis_partial_scatter` | 子轴 |
| `untouched_axis_communicated_permuted_remain` | 混合情形 |

---

## 三、这个优化为什么存在

**背景**：L4-08 讲 `reshard` → collective 的转换时，
"轴在两个维之间交换"用 `collective_permute`（L4-08 的对照表第四条）。

但**当 permute 后面紧跟着 `all_to_all`** 时，
permute 的效果可以被"`reshape` + 调整 all_to_all 的轴位置"吸收。

**一般原则**：
> **`all_to_all` 按【轴名】搬运，所以轴在源维内的顺序不影响结果。**
> 任何"只改变轴顺序"的 `collective_permute`，如果后面跟着 `all_to_all`，都是冗余的。

**这与 L4-08 的冗余 reshard 消除同类** —— 都是"消除不产生必要通信的算子"。
区别在于：
- L4-08 消除的是**完全无通信**的 reshard（前后分片相同）
- L4-10 消除的是**有通信但可被吸收**的 permute

---

## 四、13 个用例的族谱

| 族 | 文件 | 用例数 | 定义 |
|---|---|---|---|
| **全散开** | `all_to_all_fully_scattered` | 7 | 复合轴里**所有**轴都被搬走 |
| **部分散开** | `all_to_all_partially_scattered` | 6 | **只有部分**轴被搬走，其余留在原维 |

**两个文件共同的模式**：
```
collective_permute  →  reshape  →  all_to_all × N  →  reshape
     （删除）            （拆复合轴）    （按轴搬）        （合回去）
```

**一句话总结**：
> **`all_to_all` 按轴名搬运 → 只改轴序的 `collective_permute` 是冗余的。**
> 用 `reshape` 拆开复合轴、调整 `all_to_all` 的目标维，就能省下一次真实通信。
