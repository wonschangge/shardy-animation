<!-- sdy-coverage
transforms/export/test/convert_global_to_local/sdy_constant.mlir
transforms/export/test/convert_global_to_local/sdy_named_computation.mlir
transforms/export/test/convert_global_to_local/sdy_manual_computation.mlir
-->

# L5-03 · lowering-sdy-structural — 源 IR

**三个"结构算子"的降级方式** —— 它们的处理方式**各不相同**。

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/sdy_constant.mlir` | 48 | 4 |
| `convert_global_to_local/sdy_named_computation.mlir` | 58 | 2 |
| `convert_global_to_local/sdy_manual_computation.mlir` | 208 | 8 |

**合计 314 行 / 14 用例。**

网格：

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 三种降级方式总表

| 结构算子 | 降级方式 |
|---|---|
| `sdy.constant` | **按分片裁剪** —— splat 直接换类型，dense 用切片 |
| `sdy.named_computation` | **保留**，只把内部类型改成局部 |
| `sdy.manual_computation` | **直接展开** —— 区域消失，内部算子内联到外层 |

**注意**：TODOLIST 说 named_computation 是"内联"，但**实际测试显示它被保留了** ——
以实际 IR 为准。

---

## 二、`sdy.constant`：**按分片裁剪**

### 四种情形

```mlir
func.func @sharded_splat() -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
```
```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} dense<1.0> : tensor<4x4xf32>
```

```mlir
  return %0 : tensor<4x4xf32>
```

```mlir
}
```



**读法**（**情形 ①：splat + 有分片**）：
- `dense<1.0>` 是**广播常量** —— 所有元素都是 1.0。
- 局部类型 `2x1`（`4/2 × 4/4`）。
- **直接变成局部形状的常量** —— **不需要切片**！因为内容处处相同，切出来还是全 1.0。

```mlir
func.func @sharded_dense() -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
```
```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
```

```mlir
  return %0 : tensor<4x2xf32>
```

```mlir
}
```



**读法**（**情形 ②：dense + 有分片**）：
- `dense<[[1,2],[3,4],...]>` 是**稠密常量** —— 元素各不相同。
- → 用 **`partition_id`** + 查找表 `[0,0,0,0,2,2,2,2]` + `dynamic_slice`。
- **与 L5-01 / L5-02 同一套路**，但这里用的是 **`partition_id`**（L5-01 用的是 `replica_id`）。

```mlir
func.func @unsharded_splat() -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) {
```
```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}]>]>} dense<1.0> : tensor<4x4xf32>
```

```mlir
  return %0 : tensor<4x4xf32>
```

```mlir
}
```



**读法**（**情形 ③：无分片**）：
- 类型**不变**（`4x4`）—— 直接转成 `stablehlo.constant`。

```mlir
func.func @unsharded_dense() -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) {
```
```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}]>]>} dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi32>
```

```mlir
  return %0 : tensor<4x2xi32>
```

```mlir
}
```



**读法**（**情形 ④：无分片 + dense**）：
- 同样直接转换，类型不变。
- 注意 `{LITERAL}` 修饰符 —— 因为 `[[1, 2], ...]` 里的方括号是 FileCheck 特殊字符。

### ★ 三种情形的对照

| 情形 | 常量类型 | 分片 | 处理 |
|---|---|---|---|
| ① | **splat** | 有 | 直接变**局部形状**的常量（内容相同） |
| ② | **dense** | 有 | 全局常量 + `partition_id` + 切片 |
| ③ | splat | 无 | 直接转换，**类型不变** |
| ④ | dense | 无 | 直接转换，**类型不变** |

**核心洞察**：
> **splat 常量不需要切片** —— 因为所有元素相同，"切出来"和"原样"没区别。
> 只有 **dense + 有分片** 才需要 `partition_id` + 切片。

**这解释了一个常见疑问**：为什么有些分片常量降级后很"重"（六步），
有些却很"轻"（一步）—— 取决于**常量是不是 splat**。

---

## 三、`sdy.named_computation`：**保留**，只改类型

```mlir
func.func @flat(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
```
```mlir
  %0 = sdy.named_computation<"my_comp">(%arg0) in_shardings=[<@mesh_2_4, [{"x"}, {}]>] out_shardings=[<@mesh_2_4, [{"x"}, {}]>] (%arg1: tensor<16x32xf32>) {
```

```mlir
    %1 = stablehlo.tanh %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : tensor<16x32xf32>
```

```mlir
    sdy.return %1 : tensor<16x32xf32>
```

```mlir
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

```mlir
  return %0 : tensor<16x32xf32>
```

```mlir
}
```





**读法**：
- **`sdy.named_computation` 被保留** —— op 还在！
- **所有类型变成局部**：函数签名 `16x32` → `8x32`、block arg → `8x32`、
  `tanh` 的操作数 → `8x32`、`sdy.return` → `8x32`。
- **`in_shardings` / `out_shardings` 不变** —— 它们记录"怎么切的"。

**为什么保留**：
`named_computation` 是"**内联的函数**"（L1-08）——
导出期它会被 **L4-12 的 `export-named-computations`** outline 成真正的函数。
本 pass 只需把**类型**改成局部，结构留给后续 pass 处理。

### 嵌套的情形

```mlir
func.func @two_nested(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>}) {
```



**读法**：
- **两层嵌套**都被保留，且**都改成局部类型**（`2x32`）。
- `16 / (2×4) = 2` ✓（`x=2, y=4` 都切在第 0 维）
- **递归处理** —— 与 L5-01 讲的"区域算子内部同样处理"一致。

---

## 四、★ `sdy.manual_computation`：**直接展开**

```mlir
func.func @no_free_axes_two_manual_axes(%arg0 : tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"z"}]>}) {
```
```mlir
```mlir
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4_2, [{"x"}, {"z"}]>]>} : tensor<16x32xf32>
  %1 = sdy.manual_computation(%0)
    in_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    out_shardings=[<@mesh_2_4_2, [{"x"}, {"z"}]>]
    manual_axes={"x", "z"}
    (%arg1: tensor<8x16xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<8x16xf32>
    %3 = stablehlo.tanh %2  : tensor<8x16xf32>
    sdy.return %3 : tensor<8x16xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>)
  func.return %1 : tensor<16x32xf32>
}
```





**读法**（本课最重要的动作）：
- **`manual_computation` 完全消失** —— 区域被**展开**到外层。
- 区域内的 `add` + `tanh` 变成**平铺的算子**，直接读 `%[[ABS]]`。
- 结果：`abs` → `add` → `tanh` → `return`，**四行线性代码**。

**为什么可以展开**：
- `manual_axes={"x", "z"}` 表示"区域内这两个轴**不再分片**"。
- 但**导出后所有张量已经是局部的** ——
  "区域内不分片"和"外层"**是同一回事**。
- 所以区域这层"壳"**没有存在的必要**，直接去掉。

**这就是 TODOLIST 验收点问的**："能预测 `manual_computation` 展开后的形状"——
**展开后形状就是区域内的局部形状**（本例 `8x16`），区域内外**一致**。

### 8 个用例覆盖的情形

| 用例 | 场景 |
|---|---|
| `no_free_axes_two_manual_axes` | **无自由轴**、两个 manual 轴 |
| `one_free_axis_one_manual_axis` | **一个自由轴**、一个 manual 轴 |
| `nested_manual_computations` | **嵌套**的 manual_computation |
| `stablehlo_all_gather` | 区域内含 **all_gather** |
| `stablehlo_all_reduce` | 区域内含 **all_reduce** |
| `stablehlo_all_to_all` | 区域内含 **all_to_all** |
| `stablehlo_collective_permute` | 区域内含 **collective_permute** |
| `stablehlo_reduce_scatter` | 区域内含 **reduce_scatter** |

**最后 5 个用例很重要**：`manual_computation` 里**可以包含集合通信** ——
展开后它们变成**局部的通信算子**（与 L5-02 的降级配合）。

**`one_free_axis_one_manual_axis` 值得注意**：
- 有**自由轴**时，区域内该轴**仍然可以分片** ——
  所以展开时要**保留自由轴的分片信息**。
- 这与 L1-07 讲的"自由轴仍可传播"一致。

---

## 五、三个结构算子的对照

| | `sdy.constant` | `sdy.named_computation` | `sdy.manual_computation` |
|---|---|---|---|
| **降级方式** | 按分片裁剪 | **保留**，改类型 | **展开**（区域消失） |
| **为什么** | 常量内容需要"切" | 留给 L4-12 outline | 区域内外的分片状态相同 |
| **用例数** | 4 | 2 | 8 |

**一句话总结**：
> **结构算子的降级方式取决于它的语义** ——
> 常量要"切"、命名计算要"留"、手动计算要"展开"。
>
> 判断依据：**这一层结构在局部视图下还有没有意义？**
> - 常量的"内容"需要按分片调整 → **要处理**
> - 命名计算的"名字"仍然有意义 → **保留**
> - 手动计算的"区域"已经没有区别 → **展开**
