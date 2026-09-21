# L3-10 · `misc-import-cleanup` — 杂项导入清理

> 层：**L3 · 导入流水线** ｜ 优先级：P2 ｜ 前置课：`L3-06`、`L2-01`

## 学习目标

看完这一课，你应该能：

1. 说出为什么大小为 1 的轴要删、以及**网格定义为什么不变**；
2. 说出 `size-1` 轴会被从哪**四种位置**清理；
3. 说出 `pre-order-funcs` 的排序规则与目的；
4. 说出 `propagate-sharding-from-func-to-call` 的三种情形，以及它与 L3-06 的衔接。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 | pass |
|---|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/remove_size_one_axes.mlir` | 166 | 10 | `-sdy-remove-size-one-axes` |
| `shardy/dialect/sdy/transforms/import/test/pre_order_funcs.mlir` | 19 | 3 | `-sdy-pre-order-funcs` |
| `shardy/dialect/sdy/transforms/import/test/propagate_sharding_from_func_to_call.mlir` | 151 | 20 | `-sdy-propagate-sharding-from-func-to-call` |

## 场景（6 幕）

1. 三个收尾 pass 总览
2. **★ 为什么删大小为 1 的轴**
3. 四种位置都会被清理
4. `pre-order-funcs`：按调用序重排函数
5. **★ 把函数结果的分片搬到调用点**
6. 练习

## 核心结论

### ① `remove-size-one-axes`

- **网格定义不变**，只改**引用 size-1 轴的分片**。
- **为什么删**：大小为 1 的轴上只有 1 台设备，沿它分片**不会真的切开任何东西**。
  保留的代价：分片变长难读、可能产生**无意义通信**（对宽度 1 的维度做 all-gather）、
  干扰"两个分片是否等价"的判断。
- **四种位置**都会被清理：维分片 / `replicated` / `unreduced` / 算子上的分片。
  清空后属性**不再输出**（如 `unreduced` 整个消失）。
- **逐轴判断**：没有 size-1 轴的网格（`@mesh2`）上的分片**一个字符都不变**。

### ② `pre-order-funcs`

- 把"被调者在前"的书写顺序改成"**调用者在前**"的**前序遍历**顺序。
- 例：`func2, func1, main` → `main, func1, func2`。
- **目的**：让后续 pass 能**按顺序单遍处理**，不必来回跳转（调用图线性化）。

### ③ `propagate-sharding-from-func-to-call`

| 情形 | 行为 |
|---|---|
| 调用点无分片 | 从函数结果**抄过来** |
| 调用点已有分片 | **不覆盖**（保持调用点的） |
| 两者都没有 | 什么都不加 |

- **与 L3-06 的衔接**：L3-06 的 NOTE 说 `we ignore any arg/result shardings on the function`
  —— `out_shardings` 只来自调用点。若不先搬走，函数结果上的分片就会**丢失**。
- **顺序**：`-sdy-propagate-sharding-from-func-to-call` → `-sdy-import-func-calls`

## 练习

见第 6 幕。三道题分别考 size-1 轴、函数排序、分片搬迁。

## 验收点

- [x] `check_ir_fidelity.py`：23 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：三个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过（本课被它抓到 2 处笔误）
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出三个 pass 各自解决什么问题
