# L3-04 · `add-data-flow-edges` — 插入数据流边

> 层：**L3 · 导入流水线** ｜ 优先级：P1 ｜ 前置课：`L2-08`、`L3-01`

## 学习目标

看完这一课，你应该能：

1. 说出**哪些位置**需要插边（区域外结果 / 区域内块参数）；
2. 说出核心规则：**每个结果一条边**（判据是结果个数，不是使用情况）；
3. 说出两条**跳过**规则（token / 动态形状）及各自的理由；
4. 说出 `manual_computation` 的边为什么**带分片**，以及内外两侧的差异。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/add_data_flow_edges.mlir` | 253 | 16 个用例，按 6 个算子族归纳，本课选讲 6 个代表 |

RUN 行：`sdy_opt %s -sdy-add-data-flow-edges -split-input-file`

## 场景（6 幕）

1. 边是怎么被插进来的
2. **★ 核心规则：每个结果一条边**
3. 两条跳过规则
4. **★ `manual_computation` 的边带分片**
5. 16 个用例的族谱
6. 练习

## 核心结论

- **两处都要插**：区域算子的**每个结果**（区域外）+ 区域内的**每个块参数**。
  这正是 L2-08 说的「边是双向通道」的**结构基础**。
- **判据是结果个数**，不是使用情况：没被使用的结果**照样插边**
  （`while_unused_result`）。这保持「每个结果恰好一条边」的不变量。
- **边独占结果**：下游全部改用边，保证「一个结果 ↔ 一条边」的对应关系。
- **两条跳过规则**：
  - **token**（non-shaped）→ 分片无意义
  - **动态形状**（`tensor<?x?xf32>`）→ 边要求静态形状（L1-08）
  - 跳过的值**直接连到下游**，不经过边。
- **`manual_computation` 的边带分片**：因为区域算子不能带分片属性，
  `in/out_shardings` 只能通过边表达。
  - 区域内的边来自 `in_shardings`，且**去掉** `replicated`（区域内是局部世界）
  - 区域外的边完整来自 `out_shardings`
  - 对比：其它算子的边是**空的**，等传播去填。

## 用例族谱（16 个）

| 族 | 数量 | 族 | 数量 |
|---|---|---|---|
| `case` | 3 | `named_computation` | 5 |
| `optimization_barrier` | 2 | `manual_computation` | 3 |
| `while` | 1 | 综合 | 1 |

## 练习

见第 6 幕。三道题分别考结果个数、跳过规则、manual 的边。

## 验收点

- [x] `check_ir_fidelity.py`：18 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一个区域算子，能说出会插几条边、哪些值被跳过
