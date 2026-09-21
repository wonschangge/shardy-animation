# L4-11 · `per-instruction-partitioning` — 逐指令分区

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L1-07`、`L4-09`

## 学习目标

看完这一课，你应该能：

1. 说出被选中 / 未被选中的指令**分别怎么处理**；
2. **写出只分区 `dot` 与 `pad` 的 filter 串**（TODOLIST 验收点）；
3. 说出三种 `filter` 语法及其适用场景；
4. 说出这个 pass 的用途（**bisect**）。

## 覆盖的测试文件（3 个 / 665 行 / 27 用例）

| 文件 | 行数 | 用例数 | filter 语法 |
|---|---|---|---|
| `transforms/export/test/per_instruction_partitioning.mlir` | **578** | **23** | 算子名子串 |
| `transforms/export/test/per_instruction_partitioning_range.mlir` | 49 | 2 | `selectLow` / `selectHigh` |
| `transforms/export/test/per_instruction_partitioning_subroutine.mlir` | 38 | 2 | `func=<名字>` |

## 场景（6 幕）

1. **★ 核心机制：只对指定的指令跑分区器**
2. **★ 三种 `filter` 语法**
3. 位置选择：同一个函数里两个 `add`
4. 函数限定：`func=` 语法
5. `bisect`：这个 pass 为什么存在
6. 练习

## 核心结论

### 核心机制

- **选中的指令** → 包进 `sdy.manual_computation`：
  区域内用**局部形状**（全局 `8x32` 沿 `x=2` 切 → 局部 `4x32`）、不带分片属性、
  `manual_axes` 列出被冻结的轴。
- **未选中的指令** → **保持原样**，完全不受影响（用例用 `CHECK-NOT` 锁定）。

### ★ 三种 `filter` 语法

| 语法 | 写法 | 用例数 | 适用场景 |
|---|---|---|---|
| **算子名子串** | `filter=dot,pad` | 23 | 怀疑某个**算子**有问题 |
| **指令位置** | `filter='selectLow=0, selectHigh=0'` | 2 | 怀疑某个**位置**的指令 |
| **所在函数** | `filter=func=subroutine,add` | 2 | 同名算子在多处，只想测**某一个函数** |

**验收点答案**：只分区 `dot` 与 `pad` → **`filter=dot,pad`**。

**细节**：
- 算子名是**子串**匹配（写 `dot` 会匹配 `stablehlo.dot`）。
- `while` / `call` / `if` 这类**区域算子**也能被 filter 选中。
- `range` 文件的两个 RUN 行**参数顺序相反**，验证顺序不影响结果。

### bisect

**场景**：导出流水线在某段 IR 上失败，但不知道是哪个算子的问题。
**用法**：用 `filter` 逐个/分组测试，看问题在哪一组复现。
这是一个**调试工具**性质的 pass —— 不参与正常编译流程。

### 与 L1-07 / L4-09 的呼应

| 课 | 谁包 `manual_computation` | 为什么 |
|---|---|---|
| **L4-09** | permutation 因子的消解 | 表达 halo |
| **L4-11** | 指定的指令 | 表达「这个算子我已手动分好了」 |

**共同点**：`manual_computation` 是 Shardy 里表达「**区域内自己管分片**」的**通用机制**。

## 练习

见第 6 幕。三道题分别考核心机制、三种语法、bisect 用途。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能写出只分区 dot 与 pad 的 filter 串（`filter=dot,pad`）
