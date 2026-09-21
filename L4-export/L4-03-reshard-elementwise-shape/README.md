# L4-03 · `reshard-elementwise-shape` — 逐元素与形状类的 reshard

> 层：**L4 · 导出流水线** ｜ 优先级：P0 ｜ 前置课：`L2-01`、`L4-02`

## 学习目标

看完这一课，你应该能：

1. 说出逐元素算子的**三条位置规则**（"更大"的一侧优先）；
2. 说出两侧维度都不同时的**并集**解法；
3. 解释 `broadcast_in_dim` 为什么在**输入侧**按 `dims` 反向映射；
4. 解释 `reshape` 为什么**常需 reshard**（因子对应）；
5. 说出五个用例族各自的一句话规则。

## 覆盖的测试文件（9 个 / 1086 行 / 109 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/elementwise_ops.mlir` | 95 | 11 |
| `insert_explicit_reshards/broadcast_in_dim.mlir` | 21 | 2 |
| `insert_explicit_reshards/bitcast_convert.mlir` | 41 | 4 |
| `insert_explicit_reshards/reshape.mlir` | **479** | **47** |
| `insert_explicit_reshards/reverse.mlir` | 58 | 6 |
| `insert_explicit_reshards/concatenate.mlir` | 109 | 10 |
| `insert_explicit_reshards/clamp_select.mlir` | 45 | 4 |
| `insert_explicit_reshards/pad_slice.mlir` | 177 | 19 |
| `insert_explicit_reshards/dynamic_slice_dynamic_update_slice.mlir` | 61 | 6 |

## 场景（6 幕）

1. **★ 逐元素算子：reshard 插在哪一侧**
2. 两个特例：并集与双侧 reshard
3. **★ `broadcast_in_dim`：按 `dims` 反向映射**
4. `reshape`：为什么常需 reshard
5. 其余五个文件的规则速查
6. 练习

## 核心结论

- **逐元素算子三条位置规则**（统一为一句话：**算子采用"更大"的那一侧的分片；另一侧插 reshard**）：
  1. 输入无分片、结果要 → reshard 插在**之前**
  2. 输入侧"更大" → reshard 插在**之后**
  3. 输出侧"更大" → reshard 插在**之前**
- **并集解法**：两侧维度都不同时，算子被放在**两个分片的并集**上执行
  （如 `[{"x"}, {"y"}]`），这样两侧信息都不丢。该用例插 **3 条** reshard。
- **`broadcast_in_dim`**：按 `dims` **反向映射**到输入侧 —— 不是"算完再搬"，而是
  "**先摆好位置再算**"。因为 broadcast 不改变元素值，分片可以自由重排。
- **`reshape`**：看**因子对应**（不是维度）。对不上就 reshard，可能**两侧都插**。
  切的是**最 major 因子**时能对应，用 `CHECK-NOT: sdy.reshard` 锁定"一条都不插"。
- **五族速查**：
  | 族 | 用例数 | 规则 |
  |---|---|---|
  | 逐元素（`elementwise_ops` + `clamp_select`） | 15 | 对应维同分片 |
  | 广播（`broadcast_in_dim`） | 2 | 按 `dims` 反向映射 |
  | 形状变换（`reshape` + `reverse` + `bitcast_convert`） | 57 | **因子对应** |
  | 拼接（`concatenate`） | 10 | 拼接维分片必须一致 |
  | 切片填充（`pad_slice` + `dynamic_slice_...`） | 25 | 尺寸变化 → 可能不可整除 |

**一句话总结**：逐元素类看「对应维」；形状类看「因子对应」；拼接类看「拼接维一致」。

## 练习

见第 6 幕。三道题分别考位置规则、并集、reshape 的因子对应。

## 验收点

- [x] `check_ir_fidelity.py`：27 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：9 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定算子两端不同分片，能判断 reshard 插在哪一侧
