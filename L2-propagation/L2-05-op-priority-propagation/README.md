# L2-05 · `op-priority-propagation` — 算子优先级传播

> 层：**L2 · 传播算法** ｜ 优先级：P1 ｜ 前置课：`L2-01`、`L2-04`

## 学习目标

看完这一课，你应该能：

1. 说出算子优先级传播的**机制**（按算子类型分批、每批跑一遍完整激进传播）；
2. 说出方向优先级的规则（`BOTH > BACKWARD == FORWARD > NONE`）；
3. 解释「逐元素优先于 dot」为什么能改变冲突结果；
4. 说出**多使用者**为什么触发前向/反向传播的**推迟**；
5. 说出 pass-through 因子为什么优先于 reduction 因子。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/op_priority_propagation.mlir` | 313 | 15 个用例，本课选讲 6 个代表 |

RUN 行：`sdy_opt %s -split-input-file -sdy-op-priority-propagate`

## 场景（7 幕）

1. 传播金字塔第 3 层：按算子类型分批
2. 典型效果：逐元素赢过 dot
3. 多使用者：推迟前向传播
4. pass-through 因子优先于 reduction 因子
5. 方向优先级 `BOTH > FWD == BWD > NONE`
6. 两个补充观察：约束会被传播、串联 dot 逐级生效
7. 练习

## 核心结论

- **机制**：从 op-priority 0 开始逐级递增直到不动点；对优先级 `p`，考虑所有 `i < p` 的算子启发式；
  每个算子取最表达的方向；**每一轮内部都完整跑一遍激进传播**。
- **分批的实质**：它决定冲突时**谁先发声**。逐元素/广播等「直通」算子排在前面，
  dot/reduce 等「改变维度」的排在后面。
- **多使用者会推迟传播**：结果被多个下游使用时推迟前向；操作数被多个算子使用时推迟反向。
  推迟不等于不传，只是等更明确的信息。
- **方向优先级**：`BOTH > BACKWARD == FORWARD > NONE`；一轮中先见 FWD 又见 BWD 就升级为 BOTH。
- **pass-through 因子优先于 reduction 因子** —— 与「逐元素优先」同一条设计思想：
  先把便宜、无损的定下来。

## 练习

见第 7 幕。三道题分别考分批顺序、多使用者推迟、因子类型优先级。

## 验收点

- [x] `check_ir_fidelity.py`：19 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出一个冲突会由哪个批次先发声
