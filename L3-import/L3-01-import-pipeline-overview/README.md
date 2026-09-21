# L3-01 · `import-pipeline-overview` — 导入流水线总览

> 层：**L3 · 导入流水线**（开篇） ｜ 优先级：P0 ｜ 前置课：`L1-07`、`L2-08`

## 学习目标

看完这一课，你应该能：

1. 说出导入流水线的**目的**（把用户 IR 规范化成传播友好的形态）；
2. 说出 L3 层 11 课各做什么；
3. 解释**为什么 pass 顺序**（先插边、后约束）不能反；
4. 说出分片组 id 的**传递闭包合并 + 重编号**；
5. 说出 **manual 轴清理**的三种情形与统一规则。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/import_pipeline.mlir` | 165 | 8 个用例全部覆盖 |

RUN 行：`sdy_opt %s -split-input-file -sdy-import-pipeline='dedup-functions-fully=true'`

## 场景（6 幕）

1. L3 的主题：传播之前做了什么
2. 先插边，再应用约束
3. 分片组 id：传递闭包合并 + 重编号
4. manual 轴清理：三种情形
5. 函数调用：三种调用图形态
6. 练习

## 核心结论

- **导入流水线的目的**：把「用户怎么写都行」变成「传播只需处理一种形态」。
- **pass 顺序是隐式契约**：`-apply-sharding-constraints` 必须在 `-add-data_flow_edges` **之后**
  —— 否则约束会落到边的前面，传播看不到它。
- **分片组**：先做**传递闭包合并**（谁和谁间接相关），再**重编号**为 `0..N-1`。
  4 条 → 2 条、3 个 id → 1 个。
- **manual 轴清理**：`manual_axes` 里没切维度的轴 → 移进 `replicated`
  （L1-07 不变量的补齐）。三种情形：开维分片 / 全闭分片（还提升到函数参数）/
  清理必须先于插边。
- **函数调用**：三个用例锁定不同调用图（单次 / 两次 / 链式）下函数**保持原样**的行为。

## 练习

见第 6 幕。三道题分别考 pass 顺序、组规范化、manual 轴清理。

## 验收点

- [x] `check_ir_fidelity.py`：21 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出导入期至少 5 类规范化
