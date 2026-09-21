# L2-09 · `manual-computation-propagation` — 手动计算的传播

> 层：**L2 · 传播算法** ｜ 优先级：P0 ｜ 前置课：`L1-07`、`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出手动计算在传播中的角色（**不是墙，是带过滤的膜**）；
2. 说出 **in/out_shardings 是可写的**，以及自由轴被**追加**的位置规则；
3. 说出追加前为什么要**先移除已有的自由轴**；
4. 区分 manual 轴在「切维」与「replicated」两种角色；
5. 说出**闭维保护**规则。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/basic_propagation_manual_computation.mlir` | 415 | 27 个用例，按 8 组归纳，本课选讲 6 个代表 |

RUN 行：`sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-basic-propagate -sdy-sink-data-flow-edges`

## 场景（7 幕）

1. 冻结 manual 轴，但自由轴照传
2. 没有 manual 轴时：out_sharding 反向传进体内
3. **★ 核心：自由轴被追加到 in/out_shardings**
4. 追加前会先移除已有的自由轴
5. `replicated` 里的 manual 轴与自由轴
6. 闭维保护：不往里追加任何轴
7. 练习

## 核心结论

- 手动计算**不阻断传播**：`manual_axes` 里的轴被冻结，其余是自由轴，照常传播。
- **in/out_shardings 是可写的**：体内或体外的自由轴会被**追加**进去。
- **追加位置**：manual 轴始终在前，自由轴拼在后面（与 L1-07 的顺序约束一致，
  因为局部形状由 manual 轴前缀决定）。
- **追加前先移除已有的自由轴**：自由部分是**整体替换**，不是叠加
  （否则会出现同一维重复切两次）。
- `replicated` 里的 manual 轴**不影响局部形状**；自由轴也可以进 `replicated`。
- **闭维保护**：那一维是 `{"b"}`（不带 `?`）时，传播**完全不碰**它 ——
  三个用例覆盖了从体内 / out 外部 / in 外部三个方向的尝试。

## 用例族谱（27 个）

| 族 | 数量 |
|---|---|
| 基础 / 体内传播 | 5 |
| **追加**自由轴 | 7 |
| **移除**已有自由轴 | 2 |
| manual 轴组合 | 2 |
| 自由轴落位 | 2 |
| replicated | 6 |
| 闭维保护 | 4 |
| token | 1 |

## 练习

见第 7 幕。三道题分别考追加规则、移除规则、闭维保护。

## 验收点

- [x] `check_ir_fidelity.py`：19 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定初始 in_shardings 与体内自由轴，能写出最终结果
