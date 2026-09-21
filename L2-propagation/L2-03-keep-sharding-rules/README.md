# L2-03 · `keep-sharding-rules` — 保留分片规则

> 层：**L2 · 传播算法** ｜ 优先级：P2 ｜ 前置课：`L2-01`、`L1-05`

## 学习目标

看完这一课，你应该能：

1. 说出 `keep-sharding-rules=true` 的作用与它**不改变**什么；
2. 区分留在 IR 里的**两类规则**（已有的自定义规则 / 传播推导的规则）；
3. 说出这个选项在排查问题时的用法与顺序。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/basic_propagation_keep_sharding_rules.mlir` | 17 | 唯一的 1 个用例 |

RUN 行：`sdy_opt %s -sdy-basic-propagate='keep-sharding-rules=true'`

## 场景（4 幕）

1. 选项的作用
2. 用例解读：两类规则都留下
3. 什么时候该打开它
4. 练习

## 核心结论

- 这是**观测性**选项：**不改变传播结果**，只决定规则要不要留在 IR 里。
- 留下的规则有两类：
  ① 算子原本就有的（用户自定义，常含**复合因子**）；
  ② 传播**推导**出来的（为没有规则的算子补的）。
- 典型用途：确认传播用的是哪条规则、检查推导出的规则是否符合预期。
- **排查顺序**：先看规则 → 再看传播策略 → 最后才怀疑分片本身。
- 配套工具：`-sdy-populate-op-sharding-rules`（L2-10）、`-debug-sharding-origins`（L2-12）。

## 练习

见第 4 幕。

## 验收点

- [x] `check_ir_fidelity.py`：6 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出两类规则的区别与排查顺序
