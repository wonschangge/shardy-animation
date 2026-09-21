# L2-08 · `data-flow-edges` — 数据流边传播

> 层：**L2 · 传播算法** ｜ 优先级：P0 ｜ 前置课：`L1-08`、`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出区域算子（`while` / `case` / `optimization_barrier` / `manual_computation`）的
   分片**写在哪里**、以及为什么不能写在算子上；
2. 说明数据流边是**双向**通道；
3. 说出多结果时边为什么**互相独立**；
4. 说出 `manual_computation` 内部的边如何被更新；
5. 说出 token 在传播中的处理方式。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/basic_propagation_data_flow_edges.mlir` | 855 | 32 个用例，按 case / while / barrier / manual 分组，本课选讲 5 个代表 |
| `shardy/dialect/sdy/transforms/propagation/test/basic_propagation_token.mlir` | 28 | 2 个用例：token 被跳过 |

RUN 行（token 文件是三步流水线）：

```
sdy_opt %s -split-input-file -sdy-basic-propagate
sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-basic-propagate -sdy-sink-data-flow-edges
```

## 场景（6 幕）

1. 区域算子：分片不能标在它身上
2. `case`：从分支返回值传到边
3. `while`：分片穿过循环体（双向）
4. 多结果：每条边互相独立
5. 补充：手动计算内的边、token 被跳过
6. 练习

## 核心结论

- **区域算子上没有分片属性**（测试反复用 `CHECK-NOT: sdy.sharding` 断言）；
  分片写在紧随其后的 `sdy.data_flow_edge` 上。
- **为什么**：区域算子的语义由所有分支/块共同决定，一条属性表达不了「谁贡献了什么」；
  多结果时更说不清。
- 数据流边是**双向**通道：`while` 的入口 / 体内 / 出口三处分片保持一致。
- **边的粒度 = 结果的元数**：两个结果就有两条边，各自独立传播、互不污染。
- `manual_computation` 内的边会被更新，**自由轴仍由传播填充**（与 L1-07 一致）。
- **token 被静默跳过**：non-shaped 类型，不报错也不加属性；同一次调用里 tensor 正常传播。
- token 文件的流水线是：**插边 → 传播 → 收边**。

## 练习

见第 6 幕。三道题分别考边的位置、双向性、token 处理。

## 验收点

- [x] `check_ir_fidelity.py`：33 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出任意区域算子的分片写在哪里
