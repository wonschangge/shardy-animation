# L4-05 · `reshard-reduction` — 归约与排序类的 reshard

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L4-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `reduce` 的两类情形（归约维切没切）；
2. 解释 `mark-partial-result-with-unreduced-axes` 这个选项的取舍；
3. 说出 **`sort` 的被排序维为什么必须全复制**；
4. 说出 `sort` 为什么需要 `need_replication` 因子；
5. 用**统一判据**（能不能靠归约补救）区分两类算子。

## 覆盖的测试文件（4 个 / 343 行 / 22 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/reduce.mlir` | 112 | 8 |
| `insert_explicit_reshards/reduce_window_select_and_scatter.mlir` | 77 | 3 |
| `insert_explicit_reshards/sort.mlir` | 140 | 10 |
| `insert_explicit_reshards/rng_bit_generator.mlir` | 14 | 1 |

## 场景（7 幕）

1. `reduce`：归约维切没切决定一切
2. **★ 两个 RUN 行的对比**
3. **★ `sort`：被排序维必须全复制**
4. 无处可搬时：只能全复制
5. 其余两个：`reduce_window` 与 `rng`
6. **★ 判据：能不能靠归约补救**
7. 练习

## 核心结论

### `reduce`

| 情形 | 条件 | 结果 |
|---|---|---|
| 归约维**未切** | 每台设备算完整的归约结果 | 保留其他维；结果不要就 reshard 掉 |
| 归约维**被切** | 产生部分和 | `unreduced` + `all_reduce` |
| **多个归约维** | 两维都被切 | `unreduced={"x","y"}` + 一次 `all_reduce` 两个轴 |

**与 `dot` 同构**：`dot` 的「收缩维」就是这里的「归约维」。

### ★ 选项对比：`mark-partial-result-with-unreduced-axes`

`reduce.mlir` **跑两次**（用 `UNREDUCED` / `NOUNREDUCED` 两组 CHECK 前缀）：

| 选项 | `reduce` 上的分片 | `all_reduce` |
|---|---|---|
| `true` | `[{}, {}], unreduced={"x"}` | **仍然插** |
| `false` | **完全没有分片属性** | **仍然插** |

**两者都会插 `all_reduce`** —— 区别只在要不要显式标出「这是部分和」。
这是**可观测性 vs 简洁性**的取舍，与 L2-03 的 `keep-sharding-rules` 同类。

### ★ `sort`：被排序维必须全复制

- 排序维上**不能有分片** → 把分片**搬到别的维**，排序，再搬回。
- **为什么**：排序是**全局操作**，元素位置取决于整条维上的所有元素。
- **与 `reduce` 的关键区别**：
  - `reduce` 沿归约维分片还能靠 `all_reduce` 补救（部分和可归约）；
  - `sort` **不能**靠通信补救（顺序不是可归约的量）→ 必须**全复制**。
- **无处可搬时**（其它维大小都是 1）→ 只能 reshard 成全复制。
  **这就是 `need_replication` 因子的来源。**
- 规则是**逐维**的：非排序维上可以各切各的（有专门用例验证）。

### `reduce_window` 与 `rng`

- **`reduce_window`**：窗口改变维度大小 → 规则里**所有因子都是 `permutation`**；
  两个操作数必须**同分片**。
- **`rng_bit_generator`**：**状态必须全复制**（否则每台设备生成不同的序列，破坏复现性）；
  但**输出数据可以自由分片**。

### ★ 统一判据

> **这个操作的结果能不能靠「归约」合并？**
> - 能（求和、取最大）→ 用 `all_reduce`
> - 不能（顺序、随机序列）→ 必须**全复制**

**为什么顺序不能归约**：`all_reduce` 合并的是「值」，而排序缺少**跨设备比较** ——
设备 A 排 `[3,1]`、设备 B 排 `[4,2]`，合并得不到 `[1,2,3,4]`。

## 练习

见第 7 幕。三道题分别考 `reduce` 的两类情形、`sort` 的规则、统一判据。

## 验收点

- [x] `check_ir_fidelity.py`：18 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：4 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一个归约/排序算子，能说出该插 `all_reduce` 还是全复制
