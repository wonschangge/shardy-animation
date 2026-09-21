# L1-06 · `collectives` — 集合通信算子

> 层：**L1 · IR 构件** ｜ 优先级：P0 ｜ 前置课：`L1-02`、`L1-03`

## 学习目标

看完这一课，你应该能：

1. 说出 8 个集合通信算子各自的作用与分组（搬数据 / 做归约 / 管未归约状态）；
2. 读懂 `gathering_axes` 这种「逐维轴列表」参数，并**推导** `out_sharding`；
3. 读懂 `all_to_all` 的 `{轴}: 源维 -> 目标维` 参数与三类约束；
4. 解释 `reduce_scatter = all_reduce + all_slice`，以及为什么它能省一次通信；
5. 说出 `collective_permute` 的「每维分片大小不变」约束；
6. 区分子轴在通信中的**四种语义**：`exact_match` / `ignored` / `suffix_of_full` / `suffix_of_subaxis`；
7. 说出规范化的四类优化（消除 null、融合、抵消、合并链）。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/collective_parse_print.mlir` | 508 | 8 个算子、71 个实例：语法、子轴语义、跨网格 |
| `shardy/dialect/sdy/ir/test/collective_verification.mlir` | 1160 | 112 条校验错误（本课选讲 12 类根因） |
| `shardy/dialect/sdy/ir/test/collective_canonicalization.mlir` | 399 | 42 个规范化用例 |

## 八个算子

| 组 | 算子 | 作用 |
|---|---|---|
| 搬数据 | `sdy.all_gather` | 分片 → 复制 |
| 搬数据 | `sdy.all_slice` | 复制 → 分片 |
| 搬数据 | `sdy.all_to_all` | 把某维的分片搬到另一维 |
| 搬数据 | `sdy.collective_permute` | 重排 / 替换轴（每维大小不变） |
| 做归约 | `sdy.all_reduce` | 部分和 → 全和（结果复制） |
| 做归约 | `sdy.reduce_scatter` | 先归约再切片 |
| 管状态 | `sdy.sharded_to_unreduced` | 已分片的轴 → 未归约（**不通信**） |
| 管状态 | `sdy.replicated_to_unreduced` | 复制的轴 → 未归约（**不通信**） |

## 场景（10 幕）

1. 八个算子地图
2. `all_gather` / `all_slice`：一对逆操作
3. `all_to_all`：维度间搬分片
4. `all_reduce` / `reduce_scatter`：归约
5. `collective_permute`：重排与替换
6. 未归约状态的两个操作
7. 子轴四种语义
8. 校验错误（12 类根因）
9. 规范化（消除与融合）
10. 练习

## 核心结论

- **`out_sharding` 是推导结果，不是输入**。它由「操作数分片 + 通信轴」唯一确定，写错会被校验器拒绝。
  这是本课最重要的一条，也是 L4-08 的基础。
- 通信轴列表是**逐维**的：外层项数必须等于张量 rank。
- 轴的位置必须与分片一致：不能"收集"一个不在该维上的轴。
- `reduce_scatter` 的本质是 `all_reduce` 后再 `all_slice`，融合后省一次通信。
- `sharded_to_unreduced` / `replicated_to_unreduced` **不发消息**，只改状态标记；恒等式：
  `all_reduce(sharded_to_unreduced(x, a), a) == all_gather(x, a)`。
- 规范化只做「数学等价且更省」的改写。

## 练习

见第 10 幕。三道题分别考选算子、推导 `out_sharding`、判断错误类型。

## 验收点

- [x] `check_ir_fidelity.py`：44 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：三个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定输入分片与通信轴，能写出正确的 `out_sharding`
