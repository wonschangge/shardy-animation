# L2-02 · `conservative-mode` — 保守传播模式

> 层：**L2 · 传播算法** ｜ 优先级：P1 ｜ 前置课：`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出 `conservative-propagation=true` 的两条禁令（**禁止分裂轴**、**禁止不可整除**）；
2. 说出两条禁令的**粒度不同**：分裂轴是全局禁令，不可整除是逐轴过滤；
3. 判断一个 reshape / slice 用例在保守模式下的传播结果；
4. 说出保守模式的代价与它适合的场景。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/basic_propagation_conservative.mlir` | 51 | 5 个用例全部覆盖 |

RUN 行：`sdy_opt %s -split-input-file -sdy-basic-propagate="conservative-propagation=true"`

## 场景（6 幕）

1. 同一个 pass，多一个开关
2. 禁令 ①：需要拆子轴 → 完全不传
3. 禁令 ②：不可整除的轴被丢掉
4. 多轴 reshape 的三种结局
5. slice：切完就不整除了
6. 练习

## 核心结论

- 保守模式**不是另一个算法**，是给基础传播加了两条禁令。
- **禁令 ①（分裂轴）触发即整条放弃** —— 测试用 `CHECK-NOT: sdy.sharding` 断言"结果上完全没有分片"。
- **禁令 ②（不可整除）只丢掉那一个轴**，其余照传。
- 典型触发场景：`8 → 2x4` 且轴大小为 4（需拆子轴）；`16 → 4x4` 且某轴大小为 8（不可整除）。
- slice 会让维度变小，原本整除的轴可能不再整除。
- **取舍**：默认模式推得更多但可能需要 padding / 额外通信；保守模式推得更少但落地更简单。

## 练习

见第 6 幕。三道题分别考分裂轴禁令、不可整除过滤、开关对比。

## 验收点

- [x] `check_ir_fidelity.py`：20 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定 reshape/slice 用例，能判断保守模式下的传播结果
