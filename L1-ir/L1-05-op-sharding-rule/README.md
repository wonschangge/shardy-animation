# L1-05 · `op-sharding-rule` — 算子分片规则

> 层：**L1 · IR 构件** ｜ 优先级：P0 ｜ 前置课：`L1-02`、`L1-04`

## 学习目标

看完这一课，你应该能：

1. 说清分片规则的作用（把算子语义与传播算法解耦）；
2. 读懂 `([i,k],[k,j])->([i,j]) {i=8,j=16,k=8}` 的每个部分；
3. 写出 reshape 这类形状变换的**复合因子**（`[ij]`、`([ij],k)->(i,[jk])`）；
4. 说出因子符号的命名规则（`i`–`z`，之后 `z_1`、`z_2`…，复合时拼接）；
5. 区分三类特殊因子与 `pass-through`，并说明 `blocked_propagation` 为何是正交的；
6. 判断一条规则与宿主算子是否自洽（数量 / rank / 因子使用）。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/sharding_rule_parse_print.mlir` | 69 | 8 个合法规则：rank-0 / 复合因子 / >17 维符号 / 四类标注 / custom |
| `shardy/dialect/sdy/ir/test/sharding_rule_parsing_failure.mlir` | 223 | 47 条解析错误（归成符号命名、书写形式、特殊集合等组） |
| `shardy/dialect/sdy/ir/test/sharding_rule_verification.mlir` | 151 | 19 条语义校验错误 |

## 场景（9 幕）

1. 为什么需要分片规则
2. 语法三部分拆解
3. 因子就是 einsum 的下标（matmul 动画）
4. 复合因子与 reshape
5. 因子符号命名（`i`–`z` 与 `z_N`）
6. 因子分类：三类特殊 + 一类正交
7. 解析错误组（符号 / 写法）
8. 语义校验错误（映射与张量对齐）
9. 练习

## 核心结论

- **因子 = einsum 下标**。传播沿因子进行，运行时在因子空间与维度空间之间投影。
- 一条规则 = 维度映射 + 因子大小 + 可选的特殊因子标注 + 可选 `custom`。
- **复合因子**用一个方括号里的连续符号表示（`[ij]`），中间**不能有空格**。
- 符号只有 18 个（`i`–`z`）；第 19 个起用 `z_1`、`z_2`…；复合时**直接拼接**（`zz_1`、`z_8z_9z_10`）。
- 官方定义 **4 类因子**：`pass-through`（默认）+ `reduction` / `need_replication` / `permutation`（三者互斥）。
  `blocked_propagation` **正交**，可叠加，只禁止传播。
- 规则必须与宿主算子自洽：映射个数 = 操作数/结果个数，映射 rank = 张量 rank，因子不重复、不越界、都被用到。

## 练习

见第 9 幕。三道题分别考读映射、写归约规则、找错误。

## 验收点

- [x] `check_ir_fidelity.py`：40 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：三个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能为 `stablehlo.transpose` 写出一条正确的规则
