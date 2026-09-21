# L6-08 · `exec-reverse-slice` — slice / reverse 的执行

> 层：**L6 · 执行与解释器** ｜ 优先级：P1 ｜ 前置课：`L6-00`、`L5-04`、`L5-09`

## 学习目标

看完这一课，你应该能：

1. 说出 `slice` 降级的**核心判据**；
2. 说出 `comm_free` 与 `with_communication` 的**唯一区别**；
3. 说出 `reverse` 为什么**通常需要通信**；
4. 说出**步长**为什么让局部形状每台不同。

## 覆盖的测试文件（9 个）

| 族 | 文件 | 验证什么 |
|---|---|---|
| **基准** | `executable_convert_global_to_local/stablehlo_slice.mlir` | 切片维未分片 |
| **不跨界** | `executable_partitioner_pipeline/stablehlo_slice_comm_free.mlir` | 范围完整 → **无通信** |
| **跨界** | `executable_partitioner_pipeline/stablehlo_slice_with_communication.mlir` | 跨设备 → **需通信** |
| **不可整除** | `executable_partitioner_pipeline/stablehlo_slice_indivisible.mlir` | `3` 不能被 `4` 整除 |
| **全复制** | `executable_partitioner_pipeline/stablehlo_slice_replicated.mlir` | 结果 `[{}]` + 网格有未用轴 |
| **全复制（单轴）** | `executable_partitioner_pipeline/stablehlo_slice_replicated_mesh_2.mlir` | 同上，网格单轴 |
| **步长** | `executable_partitioner_pipeline/stablehlo_slice_strided.mlir` | `[1:7:2]` 局部形状不同 |
| **reverse 多维** | `executable_partitioner_pipeline/stablehlo_reverse_multi_dim_divisible.mlir` | 多维反转 + 可整除 |
| **reverse 单维** | `executable_partitioner_pipeline/stablehlo_reverse_single_dim_indivisible.mlir` | 单维反转 + 不可整除 + 轴序不同 |

## 场景（5 幕）

1. **★ 核心判据：是否改变设备之间的数据归属**
2. **★ `comm_free` vs `with_communication`**
3. 三个额外复杂度：不可整除 / 步长 / 全复制
4. **★ `reverse`：分片顺序也要反转**
5. 小结

## 核心结论

### ★ 核心判据

> **`slice` 看「切片的范围是否跨越设备边界」；`reverse` 看「是否反转了被分片的维」。**

**比 L5-04 更细一层**：L5-04 讲「作用维**未分片** → 参数不变」；
本课发现 —— **即使作用维就是分片维，只要切片范围覆盖完整，也不跨设备**。

### ★ `comm_free` vs `with_communication`

| 文件 | 切片 | 第 0 维（分片维） | 通信 |
|---|---|---|---|
| `comm_free` | `[0:4, 0:2]` | **完整** | **无** |
| `with_communication` | `[1:5, 0:4]` | 取 **1~4**（跨两台） | **需要** |

**逐设备推演 `with_communication`**：设备 0 有第 0~3 行、设备 1 有第 4~7 行；
`slice [1:5]` 要第 1~4 行 —— 设备 0 能给自己 1,2,3，但**第 4 行要向设备 1 要**
→ 必须通信。

### 三个额外复杂度

| 复杂度 | 文件 | 说明 |
|---|---|---|
| **不可整除** | `slice_indivisible` | 网格 `x=4`、取 `3` 行 → `3 % 4 ≠ 0` |
| **步长** | `slice_strided` | `[1:7:2]` 取第 1,3,5 行 → **局部形状每台不同** |
| **全复制** | `slice_replicated` ×2 | 结果 `[{}]` → 需 reshard；网格可有**未使用的轴** |

**步长的特殊性**：设备 0（第 0~3 行）选中第 1、3 行（**2 个**）；
设备 1（第 4~7 行）选中第 5 行（**1 个**）→ **每台拿多少不再是简单除法**。

### ★ `reverse`

```mlir
sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %1 = stablehlo.reverse %0, dims = [0, 2]
```

**关键问题**：反转一个**被分片**的维时，**分片的顺序也要反转**！
设备 0 原本拿第 0 段，反转后应该拿**最后一段** → 需要通信。

**为什么 `reverse` 比 `slice` 更"彻底"**：
- `slice` 只是「取一段」—— 段内顺序**不变**
- `reverse` 是「整体翻转」—— **每台设备该拿的数据都变了**

**另一个细节**：分片是 `[{"b"}, {"a"}, {"c"}]` ——
**轴顺序与维度顺序不同**（第 0 维切 `b`、第 1 维切 `a`）。

## 验收点

- [x] `check_ir_fidelity.py`：46 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：9 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 `comm_free` 与 `with_communication` 的区别
