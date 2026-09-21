# L4-01 · `export-pipeline-overview` — 导出流水线总览

> 层：**L4 · 导出流水线**（开篇） ｜ 优先级：P0 ｜ 前置课：`L1-06`、`L1-10`、`L3-04`

## 学习目标

看完这一课，你应该能：

1. 说出 L3 与 L4 的**分工**；
2. 说出导出流水线做的**五件事**；
3. 解释 `reduce` + `reshard` 为什么能**融合**成 `reduce_scatter`、省了什么；
4. 说出两条分支（默认 / 显式 collective）的差异与开关的作用。

## 覆盖的测试文件

| 文件 | 行数 | RUN 行 |
|---|---|---|
| `shardy/dialect/sdy/transforms/export/test/export_pipeline.mlir` | 146 | `-sdy-add-data-flow-edges -sdy-export-pipeline` |
| `shardy/dialect/sdy/transforms/export/test/export_pipeline_explicit_collectives.mlir` | 182 | `-sdy-export-pipeline='enable-insert-explicit-collectives=true remove-all-gather-reduce-scatter-for-cmv1=true mark-partial-result-with-unreduced-axes=true'` |

## 场景（6 幕）

1. L4 的主题：分片怎么落地
2. 文件 1：对齐边界与计算分片
3. **★ 文件 2 的核心：融合**
4. 其余机制：配对算子与未归约
5. 两条分支的对照
6. 练习

## 核心结论

- **L3 vs L4**：L3 是传播**之前**的规范化；L4 是传播**之后**的落地。
- **导出流水线五件事**：① 对齐函数边界分片 ② 为每个操作数计算分片
  ③ 处理自由轴（含不可整除 → **子轴**）④ 把 `reshard` 翻译成集合通信 ⑤ **融合**。
- **pass 顺序契约**：必须先跑 `-sdy-add-data-flow-edges`（L3-04），
  因为导出流水线内部的 `-sdy-sink-data-flow-edges` 需要边已存在。
- **★ 融合**：`reduce`（标 `unreduced`）+ `sharding_constraint` →
  **`sdy.reduce_scatter`** + `all_slice`。
  不融合需要 `all_reduce` → `all_slice` **两步通信**；融合后只需**一步**。
  —— 省下的是**跨设备通信**，代价最高的一环。
- **两条分支**：
  | | 默认 | 显式 collective |
  |---|---|---|
  | 输出 | 多为 `reshard` / 分片更新 | 显式 `reduce_scatter` / `all_slice` / `all_gather` |
  | 关注点 | 整理干净 | 翻译 + 融合 |
- **为什么有这么多开关**：不同后端需要不同的输出形态。

## 练习

见第 6 幕。三道题分别考 L3/L4 分工、融合、开关的作用。

## 验收点

- [x] `check_ir_fidelity.py`：14 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过（本课被它抓到 2 处笔误）
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出导出流水线的五个动作
