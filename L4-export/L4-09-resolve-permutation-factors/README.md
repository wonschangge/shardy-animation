# L4-09 · `resolve-permutation-factors` — 置换因子消解

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L4-03`、`L4-08`

## 学习目标

看完这一课，你应该能：

1. 说出 **permutation 因子**为什么让分片无法直接对应；
2. **说出开关 `halo exchange` 时 IR 的差别**（TODOLIST 验收点）；
3. 说出 **hop** 的含义与 `comm_free` 的边界；
4. 说出 halo 不可行时的两种情形与退路。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/resolve_permutation_factors.mlir` | **1463** | **46** |
| `transforms/export/test/resolve_permutation_factors/resolve_permutation_factors_replica_id.mlir` | — | — |

### ★ 两个 RUN 行

```mlir
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true" | FileCheck %s --check-prefixes=CHECK,HALO
```

## 场景（6 幕）

1. 问题：permutation 因子上的分片无法对应
2. **★ 两种模式：`REPL` vs `HALO`**
3. `pad`：`hop` 的概念与 `comm_free`
4. `reshape` 的 14 个用例与其余算子
5. 族谱与一句话总结
6. 练习

## 核心结论

### 问题

**permutation 因子**（L2-10 第四类）意味着**尺寸不成整数倍**：
16 个元素切 2 台、每台 8 个；卷积后变成 14 —— **14 切不成两个 8**。

典型来源：`convolution`（窗口）、`pad`/`slice`、`reshape`、`reduce_window`。

### ★ 两种模式

| | `REPL`（halo=false） | `HALO`（halo=true） |
|---|---|---|
| 做法 | reshard 成**全复制** → 算子 → reshard 回去 | 包进 `sdy.manual_computation`，交换**边界** |
| 通信量 | **大**（全量复制） | **小**（只交换 halo） |
| IR | 简单（两条 reshard） | 复杂（一个区域） |

**HALO 用 `manual_computation` 的原因**：它正是「区域内自己管分片」的机制（L1-07），
天然适合表达 halo。

### `hop` 与 `comm_free`

- **`hop`** = 需要跨**几台设备**取边界数据：单跳 / 多跳 / 双向 / 超出限制。
- **`comm_free`**：padding 量恰好是「一整块」→ **两种模式都零通信**
  （`CHECK-NOT: sdy.manual_computation`）。

### 三种特殊情形

| 情形 | 说明 |
|---|---|
| `comm_free` | 分片恰好整齐 → 两种模式都零通信 |
| `beyond_halo_limit` | hop 数超上限 → **只能退回 REPL** |
| `halo_impossible` | halo 结构上不可行 → **只能退回 REPL** |

### `replica_id` 文件

`stablehlo.replica_id` 返回设备副本编号 —— 让程序**依赖设备身份**。
而分片恰好改变「哪台设备拿哪片数据」，也就改变了 `replica_id` 的语义。
**与纯数据并行的分片模型有本质冲突**，所以单独一个文件处理。

### 一句话总结

> permutation 因子上的分片无法直接对应。两条出路：
> **全复制**（REPL，简单但通信大）或 **halo exchange**（HALO，通信小但受 hop 限制）。

## 练习

见第 6 幕。三道题分别考 permutation 的问题、两种模式、hop 限制。

## 验收点

- [x] `check_ir_fidelity.py`：11 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出开关 halo exchange 时 IR 的差别
