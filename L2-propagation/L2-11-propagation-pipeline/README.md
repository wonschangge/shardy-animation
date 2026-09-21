# L2-11 · `propagation-pipeline` — 传播完整流水线

> 层：**L2 · 传播算法** ｜ 优先级：P0 ｜ 前置课：`L1-10`、`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出完整流水线做了哪 7 步；
2. 解释一个常量为什么会被**复制成多份**；
3. 说出 `sharding_constraint` 的**归宿**；
4. 说出 `unreduced` 标注如何引出 `all_reduce` + `reshard` 两条通信；
5. 说出本文件与 L2-08 / L2-09 用例的**对照关系**。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/propagation_pipeline.mlir` | 1200 | 58 个用例，整个文件**只有一条 RUN 行**；本课选讲 5 组代表 |

RUN 行：`sdy_opt %s -split-input-file -sdy-propagation-pipeline`

## 场景（6 幕）

1. 完整流水线：各层叠加后的样子
2. **★ 一个常量变三个**
3. 约束 → `reshard`：用户意图被翻译成通信
4. **★ 一个标注，两条通信**
5. 同一批用例，换个流水线再跑一遍
6. 练习

## 核心结论

- 流水线 7 步：**拆常量 → 规范化网格 → 组合并 → 逐层传播 → 插集合通信 → 约束→reshard → 清理**。
- **常量拆分**：一个常量被多处使用且需要不同分片时，复制成多份
  （前提是 `sdy.constant` 不带 folder，见 L1-10）。
- **`sharding_constraint` 在结果里不存在**，被换成 `sdy.reshard`；且只作用于它那个使用者。
- **`unreduced` 标注 → 两条通信**：`all_reduce`（归约部分和）+ `reshard`（调整分片）。
- 本文件的 `case_*` / `while_*` / `manual_computation_*` 与 L2-08 / L2-09 **同名同形**，
  只换了 RUN 行 —— 可做**对照实验**。

## 练习

见第 6 幕。三道题分别考常量拆分、约束的归宿、未归约轴的处理。

## 验收点

- [x] `check_ir_fidelity.py`：14 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一段带标注的 IR，能说出流水线会补出哪些算子
