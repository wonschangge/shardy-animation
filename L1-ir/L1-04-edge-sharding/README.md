# L1-04 · `edge-sharding` — 边分片与传播调试信息

> 层：**L1 · IR 构件** ｜ 优先级：P1 ｜ 前置课：`L1-03`

## 学习目标

看完这一课，你应该能：

1. 说清 `sdy.propagation_edges` 记录的是什么（谁在第几步把哪条轴传给了谁）；
2. 读懂四层嵌套语法：属性 → step → 轴条目 → source/targets；
3. 区分三个属性名（`propagation_edges` / `result_` / `block_arg_`）及各自的类型要求；
4. 判断 step 索引、值引用索引是否合法；
5. 明白为什么 `sdy.func_data_flow_edge` 上越界 operand 是**允许**的。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/edge_sharding_parse_print.mlir` | 14 | 属性语法与打印形式 |
| `shardy/dialect/sdy/ir/test/edge_sharding_verification.mlir` | 164 | 15 类校验用例（14 报错 + 1 反例） |

## 场景（7 幕）

1. 为什么需要边分片（分片从哪来）
2. 语法拆解（四层结构）
3. 三个属性名与类型要求
4. 错误组 1：step 索引（重复 / 负数）
5. 错误组 2：source 与 target（自环 / 重复 / 越界）
6. 错误组 3：类型 / 引用 / 轴 + **反例**
7. 练习

## 核心结论

- 边分片是**调试元数据**，不参与语义；导出前由 `-sdy-remove-propagation-debug-info` 清除。
- 结构：`{step-N = [{轴 = source -> [targets]}]}`，一个源可流向多个目标。
- `step-N`：N ≥ 0 且同一属性内不重复。
- 值引用索引必须落在宿主算子的范围内；**函数级数据流边例外**。
- `sdy.propagation_edges` 是 `PropagationEdgesAttr`；`result_` / `block_arg_` 版是 `ArrayAttr<PropagationEdgesAttr>`。
- 边分片必须同时带 `sharding`（它描述的是"某个分片"的传播路径）。

## 练习

见第 7 幕。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 `{step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]}` 的含义
