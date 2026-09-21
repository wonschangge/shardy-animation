# L1-03 · `tensor-sharding-verification` — 分片校验不变量

> 层：**L1 · IR 构件** ｜ 优先级：P0 ｜ 前置课：`L1-02`

## 学习目标

看完这一课，你应该能：

1. 区分**解析错误**（`failed to parse`）与**校验错误**（语义非法）；
2. 说出分片属性的**七条不变量**，并把任意一条报错归到其中一条；
3. 判断子轴 `(pre)size` 是否合法（取值范围 + "必须最简"）；
4. 解释为什么 `replicated` / `unreduced` 必须按网格顺序排列；
5. 处理 token、无 rank 张量、maximal-sharding 网格这些特殊类型；
6. 理解 `manual_computation` 对轴的"绑定"如何影响区域内分片。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/tensor_sharding_verification.mlir` | 556 | L1 最大单文件；33 类校验错误，按七条不变量归组 |

## 七条不变量

| # | 不变量 | 典型报错 | 本课幕 |
|---|---|---|---|
| 1 | 轴引用必须**存在且唯一** | `duplicate axis ref` / `unknown axis name` / `overlapping sub-axes` | 3 |
| 2 | 子轴参数必须**合法且最简** | `pre-size must be at least 1` / `can be merged` | 4 |
| 3 | `replicated`/`unreduced` 必须**按网格序** | `not ordered w.r.t. mesh` | 5 |
| 4 | 维分片数必须**等于 rank** | `doesn't match tensor rank: 2 != 1` | 5, 7 |
| 5 | 空的闭维**不能带优先级** | `empty and closed but has a priority` | 5 |
| 6 | 属性**类型与数量**必须匹配 | `should have ... TensorShardingPerValueAttr` | 6 |
| 7 | 上下文约束：轴**未被父级绑定** | `already bound by a parent sdy.manual_computation op` | 8 |

## 场景（9 幕）

1. 两级检查：parse vs verify
2. 七条不变量总览
3. 不变量 1：轴引用存在且唯一
4. 不变量 2：子轴参数合法性
5. 不变量 3/4/5：排序、rank、空闭维
6. 不变量 6：属性类型与数量
7. 特殊类型：token / 无 rank / maximal 网格
8. 不变量 7：manual_computation 的轴绑定
9. 练习

## 核心结论

- **读报错的诀窍**：先看它说哪条不变量，再看自己违反了哪个前提。
- 轴是"独占资源"：切维度 / 显式复制 / 未归约 —— 三选一，且只能选一次。
- 子轴要"写满"：相邻能合并就必须合并，覆盖整轴就该用完整轴。
- 排序不是语义要求，而是**规范化**要求。
- 唯一一条依赖**外部上下文**的不变量是第 7 条。

## 练习

见第 9 幕。三道题分别考轴唯一性、子轴参数、rank 一致性。

## 验收点

- [x] `check_ir_fidelity.py`：38 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一条校验报错，能说出它属于七条中的哪一条
