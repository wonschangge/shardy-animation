# L4-04 · `reshard-matmul-conv` — 矩阵与卷积类的 reshard

> 层：**L4 · 导出流水线** ｜ 优先级：P0 ｜ 前置课：`L2-10`、`L4-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `dot` 的**三类情形**（零通信 / all_reduce / reduce-scatter 模式）；
2. 说出 **reduce-scatter 模式**的三步及其与 L4-01 融合的关系；
3. 解释 `batch_group_count` 为什么让批维**缩小**；
4. 说出 `fft` / `cholesky` 的**共同结构**（受限维 + 自由维）。

## 覆盖的测试文件（4 个 / 1072 行 / 95 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/dot_dot_general.mlir` | **736** | **66** |
| `insert_explicit_reshards/convolution.mlir` | 60 | 3 |
| `insert_explicit_reshards/fft.mlir` | 74 | 7 |
| `insert_explicit_reshards/cholesky_triangular_solve.mlir` | 202 | 19 |

## 场景（6 幕）

1. **★ `dot` 的三类情形**
2. 66 个用例的命名规律
3. `convolution`：批维分组会改变大小
4. `fft` 与 `cholesky`：受限维 vs 自由维
5. 4 个文件 / 95 个用例的族谱
6. 练习

## 核心结论

### `dot` 的三类情形

| 情形 | 条件 | 结果 |
|---|---|---|
| ① **零通信** | 收缩维未切，结果与操作数一致 | 什么都不插 |
| ② **all_reduce** | 收缩维被切（两侧一致） | `unreduced` + `all_reduce` |
| ③ **reduce-scatter 模式** | 收缩维被切 **且结果也要该轴** | `unreduced` + `all_reduce` + `reshard` |

- 测试注释直接点明第 ③ 类：`This is a reduce-scatter pattern.`
- **与 L4-01 的衔接**：导出流水线会把第 ③ 类的三步**融合成一条 `reduce_scatter`**。
  本课看到的是**融合前**的形态。

### 66 个用例的冲突类型（8 组）

`compatible` / lhs vs rhs 冲突 / 子轴无重叠 / 同轴两次 /
**全相同却不兼容** / 同因子两用 / reduce-scatter / 轴序不兼容。

最反直觉的一条：`all_same_shardings` —— 分片**全相同**却仍然不兼容（语义层面冲突）。

### `convolution`

- 基础情形：输入批维切 `x`、结果无分片 → 算完 reshard 掉。
- `batch_group_count`：输入批维是**复合因子 `ij`**（`i=4` 分组 × `j=2` 输出批维），
  输入 8 → 输出 2，**批维缩小 4 倍**。
- 这就是 L2-10 那条卷积规则（`reduction={l,n,o} permutation={k,m}`）的实际作用。
- 输入切 `x`、结果要 `y` → reshard 到**并集** `[{"x", "y"}, ...]`（与 L4-03 一致）。

### `fft` 与 `cholesky` 的共同结构

**一个受限维 + 一个自由维**：
- `fft`：**变换维受限**（FFT 是全局操作）、批维自由；截断/扩展会改尺寸。
- `cholesky`：**分解维受限**（要看到整个矩阵、三角求解有依赖）、批维自由；
  19 个用例是"两维度 × 切/不切/切哪个轴"的组合。

这与 `dot` 的「收缩维 vs 非收缩维」**是同一个模式**。

**一句话总结**：收缩/变换/分解维是「受限维」，批维是「自由维」；
冲突判定就是看这两类维度上的分片是否兼容。

## 练习

见第 6 幕。三道题分别考三类情形、reduce-scatter 模式、受限维。

## 验收点

- [x] `check_ir_fidelity.py`：15 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：4 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定 dot 的 lhs/rhs/结果分片，能说出会插什么
