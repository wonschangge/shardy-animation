# L3-11 · `gather-scatter-batching` — gather/scatter 批维显式化（L3 收官）

> 层：**L3 · 导入流水线** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L3-01`

## 学习目标

看完这一课，你应该能：

1. 说出这个变换**做了什么**（隐式批维 → 显式批维）；
2. **解释隐式批维为什么导致分片规则无法表达**（本课核心）；
3. 说出五个**不转换**的情形及其原因；
4. 说出为什么需要**可执行验证**而不只是比对 IR 文本。

## 覆盖的测试文件

| 文件 | 行数 | 作用 |
|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/explicit_gather_scatter_batching.mlir` | 341 | 主测试：5 转换 + 5 不转换 |
| `executable_explicit_gather_scatter_batching/gather_iota_concat_batch_dim.mlir` | — | 可执行验证 |
| `.../gather_iota_reshaped_concat.mlir` | — | 可执行验证 |
| `.../gather_iota_at_end_of_concat.mlir` | — | 可执行验证 |
| `.../gather_iota_broadcast_concat.mlir` | — | 可执行验证 |
| `.../scatter_batch_dim.mlir` | — | 可执行验证 |

RUN 行：`sdy_opt %s -sdy-explicit-gather-scatter-batching | FileCheck %s`

## 场景（6 幕）

1. 问题：隐式批维让分片规则无法表达
2. **★ 为什么隐式批维无法分片**
3. 五个不转换的情形（同样重要）
4. 可执行验证：为什么要跑一遍
5. L3 收官：导入流水线全图
6. 练习

## 核心结论

- **背景**：JAX 里 `arr.at[jnp.arange(B), offset]` 这类逐行索引会生成「`iota` + `concat`」构造索引。
- **变换内容**：
  | 属性 | 转换前 | 转换后 |
  |---|---|---|
  | `start_index_map` | `[0, 1]` | **`[1]`** |
  | `operand_batching_dims` | （无） | **`[0]`** |
  | `start_indices_batching_dims` | （无） | **`[0]`** |
  语义完全不变，只是换了表达方式。
- **为什么隐式批维无法分片**（四步）：
  ① 写在 `start_index_map` 里 → 落在 `blocked_propagation` 因子上；
  ② 该标注禁止传播沿它推导分片；
  ③ 但这一维语义上**明明可以分片**（每个输出元素独立取一行）；
  ④ 结果：**想切却切不了**。
  **一句话：隐式批维把"可并行的批维"伪装成了"不可传播的索引维"。**
- **五个不转换**：没有 iota / 批大小不匹配 / 已是显式批维 / `index_vector_dim` 为 1 /
  dim 0 未在 `collapsed_slice_dims` 里。**5 转换 + 5 不转换** —— 因为改写语义属性的变换，
  前提条件与转换本身同等重要。
- **可执行验证**：属性改错了 IR 仍能通过校验但**数值会错**，所以 5 个
  `executable_*` 用例用 SDY 解释器真正执行，与 `part2.mlir` 的期望结果对比。

## 练习

见第 6 幕。三道题分别考变换内容、失败原因、验证方式。

## 验收点

- [x] `check_ir_fidelity.py`：6 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：六个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能解释隐式批维为什么让分片规则无法表达
