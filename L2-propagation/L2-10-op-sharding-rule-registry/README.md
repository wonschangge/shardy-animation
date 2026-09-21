# L2-10 · `op-sharding-rule-registry` — 算子分片规则注册表

> 层：**L2 · 传播算法** ｜ 优先级：P0 ｜ 前置课：`L1-05`、`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出这个 pass 的作用（**把注册表推导出的规则打印出来**）；
2. 读懂逐元素 / 广播 / 形状变换三类规则；
3. 说出 **`reduction`** 的判据与含义（切了要通信）；
4. 说出 **`permutation`** 的判据（两侧尺寸不成整数倍）；
5. 说出 `gather` 里 `need_replication` 与 `blocked_propagation` 的作用与区别；
6. 说出 `custom_call` 的内置注册与用户自定义（`, custom` 标记）。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/transforms/propagation/test/op_sharding_rule_registry.mlir` | 1213 | **本计划第二大文件**；125 个用例按 10 组归纳，本课选讲 6 组代表 |

RUN 行：`sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics`

## 场景（7 幕）

1. 注册表：为每个算子推导规则
2. 基础三族：逐元素 / 广播 / 形状变换
3. **★ `reduction`：切了就要通信的因子**
4. `permutation`：尺寸不成整数倍的因子
5. `gather`：四类标注同时出现
6. `custom_call`：内置注册 vs 用户自定义
7. 练习

## 核心结论

- 这个 pass 把规则**打印出来** —— 想知道某算子的规则，写个最小用例跑一遍即可
  （L2-03 说的「排查先看规则」的具体操作方式）。
- **逐元素**：因子一一对应，无标注。
- **广播**：结果侧多出因子（输入侧没有它）。
- **`reduction` 判据**：因子在操作数侧出现、结果侧消失 → 切了会产生部分和 → 需要 all-reduce。
- **`permutation` 判据**：因子在两侧的尺寸**不成整数倍**（卷积步长、pad 边界）。
  **转置不需要**（只重排维序，大小不变）。
- **`gather`** 同时用上 `reduction` / `need_replication` / `blocked_propagation`；
  `blocked_propagation` 是**正交标注**，与四类因子分类并列。
- **`custom_call`**：注册表内置 22 个常用算子的规则；用户自定义规则末尾带 `, custom` 标记来源。

## 用例族谱（125 个）

| 族 | 数量 | 族 | 数量 |
|---|---|---|---|
| 逐元素 / 标量 | ~12 | 索引类（gather/scatter） | 12 |
| 广播 | 6 | 三角 / 分解 / fft | ~14 |
| 点积 | 7 | `custom_call` | 23 |
| 规约 | 8 | concat / pad / all_* / 其它 | ~14 |
| 卷积 | 7 | | |
| 形状变换 | ~22 | | |

## 练习

见第 7 幕。三道题分别考 `reduction` 判据、`permutation` 判据、`custom` 标记。

## 验收点

- [x] `check_ir_fidelity.py`：32 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：源文件被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一条规则，能指出哪些因子是 reduction / permutation
