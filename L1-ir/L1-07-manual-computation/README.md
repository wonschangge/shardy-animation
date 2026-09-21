# L1-07 · `manual-computation` — 手动计算

> 层：**L1 · IR 构件** ｜ 优先级：P1 ｜ 前置课：`L1-02`、`L1-03`

## 学习目标

看完这一课，你应该能：

1. 说清 `sdy.manual_computation` 的用途（在传播之外开一个「我自己来」的口子）；
2. **算出区域内任意张量的局部形状**（外层形状 ÷ manual 轴大小）；
3. 区分 **manual 轴**（冻结）与 **自由轴**（仍归传播管），并说出顺序约束；
4. 正确嵌套多层手动计算（每层用不同的 manual 轴）；
5. 说出四类校验根因：网格一致 / 结构与数量 / 局部形状 / 轴的角色；
6. 说出规范化会消除哪几类「没用的手动计算」。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/manual_computation_parse_print.mlir` | 246 | 15 个合法用例：空体、局部形状、自由轴、嵌套、动态、token |
| `shardy/dialect/sdy/ir/test/manual_computation_verification.mlir` | 296 | 21 条不变量校验错误（本课选讲 8 类） |
| `shardy/dialect/sdy/ir/test/manual_computation_canonicalization.mlir` | 89 | 7 个规范化用例 |

## 场景（8 幕）

1. 为什么要手动计算
2. 语法四要素拆解
3. **★ 核心：局部形状怎么算**
4. 自由轴：manual 之外的轴仍归传播管
5. 嵌套：每层绑定不同的 manual 轴
6. 校验错误（四类根因）
7. 规范化：消除没用的手动
8. 练习

## 核心结论

- **局部形状 = 外层维度 ÷ ∏(该维上属于 manual_axes 的轴大小)**。自由轴**不影响**局部形状。
- `manual_axes` 里的轴必须在**所有** in/out 分片里显式出现（切维度或进 `replicated`）。
- 同一维分片里，**manual 轴必须排在自由轴之前**（更 major）。
- manual 轴必须**整除**维度大小 —— 手动计算不允许 padding。
- 嵌套时每层必须用**不同**的轴；重复绑定会被父级检查拦下。
- 区域是 `IsolatedFromAbove`：只能使用块参数。
- 规范化会消除空体、无 manual 轴、冗余 manual 轴、未使用参数等情形。

## 练习

见第 8 幕。三道题分别考局部形状计算、轴顺序、嵌套约束。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：三个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定外层形状 + manual_axes + 分片，能写出体内块参数类型
