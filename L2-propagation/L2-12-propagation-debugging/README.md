# L2-12 · `propagation-debugging` — 传播调试（L2 收官）

> 层：**L2 · 传播算法** ｜ 优先级：P0 ★ 实用课 ｜ 前置课：`L2-03`、`L2-05`

## 学习目标

看完这一课，你应该能：

1. 说出两个调试属性各自回答什么问题；
2. 读懂 `sdy.sharding_origins` 的三种来源取值（`self` / `input: N` / `output: N`）；
3. 从一个可疑的轴出发**沿来源回溯**到用户标注；
4. 读懂 `sdy.propagation_edges` 的 `step-N` 与"一对多"结构；
5. 按五步流程独立排查传播问题。

## 覆盖的测试文件

| 文件 | 行数 | 调试属性 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/debugging/test/sharding_origins.mlir` | 758 | `sdy.sharding_origins` |
| `shardy/dialect/sdy/transforms/propagation/debugging/test/edge_shardings.mlir` | 402 | `sdy.propagation_edges` |

两个文件的 RUN 行都是**三步流水线**，且每一步都打开同一个调试开关
（`-sdy-apply-sharding-constraints` / `-sdy-aggressive-propagate` / `-sdy-sink-data-flow-edges`）。

## 场景（6 幕）

1. 两个调试属性：来源与路径
2. `sharding_origins` 的三种来源
3. **★ 动手追溯：点开每个张量看它的分片从哪来**（可点击）
4. `propagation_edges`：路径与轮次
5. 实战：五步排查流程
6. 练习

## 核心结论

- **`sdy.sharding_origins`** 回答「这个轴的最终值**来自哪里**」—— 结论/快照，逐轴一个来源：
  - `"self"` —— 这个张量自己的标注（用户意图）
  - `"input: N"` —— 第 N 个输入/操作数（**前向**传播）
  - `"output: N"` —— 第 N 个输出/结果（**反向**传播）
  - 格式：函数参数/结果是**字典**；算子上是**列表**（多结果每项一个）。
- **`sdy.propagation_edges`** 回答「分片**沿哪条路径**流动过」—— 过程/轨迹：
  - 格式 `#sdy.propagation_edges<[{step-N = [{"轴" = 来源 -> [目标]}]}]>`
  - **一对多**：一次传播可同时影响多个目标
  - `step-N` 记录**轮次**
- **五步排查流程**：① 看规则 → ② 看来源 → ③ 往前追 → ④ 看路径 → ⑤ 换策略对比。

## 练习

见第 6 幕。三道题分别考三种来源、追溯方法、两个属性的分工。

## 验收点

- [x] `check_ir_fidelity.py`：9 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用（第 3 幕可点击）
- 自检：给定一个张量的分片，能用调试属性说出它被哪个输入"传染"
