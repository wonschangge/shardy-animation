# L1-02 · `tensor-sharding-syntax` — 张量分片属性语法

> 层：**L1 · IR 构件** ｜ 优先级：P0 ｜ 前置课：`L1-01`

## 学习目标

看完这一课，你应该能：

1. 区分 `sdy.sharding`（函数上）与 `#sdy.sharding_per_value`（算子上，多一层列表）；
2. 说出分片属性的四个组成部分：网格引用、每维分片、复制轴、未归约轴；
3. 读懂开维 `?`、子轴 `"b":(2)2`、优先级 `p0/p1/p2` 的写法与含义；
4. 判断内联网格何时会被规范化（iota 省略）；
5. 区分**解析错误**（`failed to parse`）与**校验错误**（语法对但语义非法）。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/tensor_sharding_parse_print.mlir` | 235 | 合法写法：两种属性形式、开闭维、replicated/unreduced、内联网格、子轴、优先级、特殊类型 |
| `shardy/dialect/sdy/ir/test/tensor_sharding_parsing_failure.mlir` | 177 | 15 类解析错误，按根因归成 3 组 |

## 场景（10 幕）

| # | 主题 | 学到什么 |
|---|---|---|
| 1 | 两种属性形式 | `sharding` vs `sharding_per_value` |
| 2 | 语法全貌 | 四个组成部分 |
| 3 | 开维与闭维 | `?` 的作用 |
| 4 | replicated / unreduced | 含 `max` 与打印顺序规范化 |
| 5 | 内联网格 vs 符号引用 | iota 省略规则同样适用 |
| 6 | 子轴 `(pre)size` | 拆解、合法与非法组合 |
| 7 | 优先级 | `pN` 的位置与传播轮次 |
| 8 | 四种边角情况 | rank-0 / 动态形状 / tuple / 无结果算子 |
| 9 | 15 类解析错误 | 按根因分 3 组 |
| 10 | 练习 | 属性解读 / 语法纠错 / 规则判断 |

## 核心结论

- **函数用 `sharding`，算子用 `sharding_per_value`**；后者是前者的列表，一项对应一个值。
- 列表项数 = **值的个数**；每项的维分片数 = **该值的 rank**（rank-0 就是 `[]`）。
- 四个部分中后三个可省略；省略 ≠ 不存在：**没被用到的轴 = 隐式复制**。
- 优先级 `pN` 写在**花括号外**：`{"a"}p0` ✓，`{"a"p0}` ✗。
- 打印顺序恒为 `replicated` 在前、`unreduced` 在后，**与输入顺序无关**。
- 解析错误（`failed to parse`）与校验错误（L1-03）是两级不同的检查。

## 练习

见第 10 幕。

## 验收点

- [x] `check_ir_fidelity.py`：50 个 IR 块全部逐字来自上述两个文件
- [x] `check_coverage.py`：两个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 `[{"a"}p0, {"b", ?}]` 每个维度的开闭与优先级
