# L1-08 · `named-computation-and-dataflow` — 命名计算与数据流边

> 层：**L1 · IR 构件** ｜ 优先级：P1 ｜ 前置课：`L1-03`、`L1-07`

## 学习目标

看完这一课，你应该能：

1. 说清 `sdy.named_computation` 解决什么问题（让传播穿过函数调用）；
2. 说出它与 `sdy.manual_computation` 的**关键区别**（冻结轴 vs 不冻结）；
3. 描述一条数据流边的三要素：**sources / targets / owner**；
4. 说出 `data_flow_edge` 的三条约束与 `func_data_flow_edge` 的两条约束；
5. 解释 owner 为什么必要（避免冗余与不一致）；
6. 把这三类错误归到「结构对齐 / 分片合法 / 边良定义」三组。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/named_computation_parse_print.mlir` | 49 | 4 种形态：单入单出、多入多出、带 in/out 分片、token |
| `shardy/dialect/sdy/ir/test/named_computation_verification.mlir` | 89 | 9 条校验错误（结构对齐 + 分片合法） |
| `shardy/dialect/sdy/ir/test/data_flow_edge_verification.mlir` | 42 | 4 条校验错误（边良定义） |
| `shardy/dialect/sdy/ir/test/func_data_flow_edge_verification.mlir` | 58 | 2 条校验错误（边良定义） |

## 场景（7 幕）

1. 传播怎么穿过函数调用
2. `named_computation` 的四种形态
3. **数据流边模型**：sources / targets / owner
4. 数据流边的三条约束
5. 函数级的边 `func_data_flow_edge`
6. 三类校验错误速查
7. 练习

## 核心结论

- `sdy.named_computation` 把**函数体内联**进一个命名区域，让传播不必理解调用语义。
  导入阶段由 `-sdy-import-func-calls` 生成，导出阶段由 `-sdy-export-named-computations` 还原。
- **与 `manual_computation` 的关键区别**：named 版**没有** `manual_axes`，区域内仍是**全局形状**。
- 一条数据流边 = **sources + targets + owner**；owner 是 targets 之一，分片属性挂在它身上。
- 传播把数据流边当作**恒等规则**处理 —— 所以它**不需要 sharding rule**。
- 边良定义的约束：结果必须**静态形状**、输入/操作数必须**单一使用者**、
  `data_flow_edge` 的输入**不能由 SDY 算子定义**（防止处理顺序回环）。
- 边上的分片校验与普通张量分片**完全复用**。

## 练习

见第 7 幕。三道题分别考与 manual 的区别、边的约束、owner 的作用。

## 验收点

- [x] `check_ir_fidelity.py`：18 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：四个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 `named_computation` 与 `manual_computation` 的三点区别
