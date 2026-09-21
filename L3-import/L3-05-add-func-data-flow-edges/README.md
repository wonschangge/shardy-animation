# L3-05 · `add-func-data-flow-edges` — 插入函数级数据流边

> 层：**L3 · 导入流水线** ｜ 优先级：P1 ｜ 前置课：`L1-08`、`L3-04`

## 学习目标

看完这一课，你应该能：

1. 说出函数级边的**两个插入位置**；
2. 说出插边数量的**判据**（值的个数，不是使用情况）；
3. 解释链式调用图里为什么**两种边共存**；
4. 说出 token 跳过规则及其理由；
5. 区分 SDY 的**三种桥接结构**。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/add_func_data_flow_edges.mlir` | 471 | 17 个用例，本课选讲 5 组代表 |

RUN 行：`sdy_opt %s -split-input-file -sdy-add-func-data-flow-edges`

## 场景（6 幕）

1. 函数级边：两个插入位置
2. 多结果 / 多使用者：与算子级边同一套规则
3. 链式调用：两种边共存
4. token 跳过（测试注释写明了理由）
5. 三种边的对照
6. 练习

## 核心结论

- **两个插入位置**：
  ① **函数参数**（函数体内）—— 每个参数一条边，函数体使用者改用边
  ② **调用结果**（调用点）—— 每个结果一条边，下游改用边
  两处合起来桥接了 L1-08 的两对关系：**调用实参 ↔ 函数形参**、**函数 return ↔ 调用结果**。
- **判据是值的个数**，不是使用情况；**多个使用者共用同一条边**（与 L3-04 一致）。
- **链式调用**：中间的 `@bar` 里**同时**有参数边与调用结果边，两者独立。
- **token 跳过**：测试注释原文 ——
  `Tokens are not static-shaped types, so no func_data_flow_edge should be created for them.`
  两个用例用 `CHECK-NOT` 精确断言「没有边」。
- **三种桥接结构**：
  | 结构 | 桥接什么 | 讲在 |
  |---|---|---|
  | `sdy.data_flow_edge` | **区域**边界 | L1-08 / L2-08 / L3-04 |
  | `sdy.func_data_flow_edge` | **函数**边界 | L1-08 / **本课** |
  | `sdy.named_computation` | **消除**函数边界 | L1-08 / L3-06 |

## 练习

见第 6 幕。三道题分别考插入位置、数量判据、token 规则。

## 验收点

- [x] `check_ir_fidelity.py`：20 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一个函数签名，能说出插几条边、插在哪
