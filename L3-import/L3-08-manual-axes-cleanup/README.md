# L3-08 · `manual-axes-cleanup` — 清理 manual 轴

> 层：**L3 · 导入流水线** ｜ 优先级：P2 ｜ 前置课：`L1-07`、`L3-01`

## 学习目标

看完这一课，你应该能：

1. 说出这个 pass 的**三个动作**；
2. 说出补 `replicated` 的三种情形（新增 / 追加 / 取并集）；
3. 说出排序的**依据**（网格声明顺序，**不是**字母序）；
4. 解释空网格为什么必须先被替换；
5. 说出三个边界（token / 子轴 / `unreduced`）与唯一的失败情形。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/manual_axes_cleanup.mlir` | 191 | 12 |
| `shardy/dialect/sdy/transforms/import/test/manual_axes_cleanup_failures.mlir` | 10 | 1 |

RUN 行：`sdy_opt %s -split-input-file -sdy-manual-axes-cleanup`

## 场景（7 幕）

1. 三个动作：补 replicated、排序、换空网格
2. **★ 动作 ①：补 `replicated`**
3. **★ 动作 ②：按网格声明顺序排序**
4. 动作 ③：空网格被替换成实际网格
5. 三个边界：token / 子轴 / `unreduced`
6. 唯一会报错的情形
7. 练习

## 核心结论

- **三个动作**（顺序：换空网格 → 补 replicated → 排序）：
  1. **补 `replicated`**：`manual_axes` 里没切维度的轴加进去。三种情形：
     **新增** / **追加到已有** / **取并集**（in 与 out 各缺一个时都补成完整并集）。
  2. **排序**：按**网格声明顺序**（`@mesh = <["c"=2, "a"=2, "b"=2]>` → `{"c","a","b"}`）。
     **不是字母序** —— 用另一个网格验证即可确认。
  3. **换空网格**：`@empty_mesh` 替换成同一个手动计算里实际使用的网格
     （替换后才能知道"轴有哪些"，才能补 replicated 与排序）。
- **三个边界**：
  - **token** 的 `<@mesh, []>` 保持原样（rank-0 不能有 `replicated`，L1-02）
  - **子轴**被展开成**完整子轴集合**（逐子轴决定，不能笼统加一个轴名）
  - **`unreduced`** 原样保留（分片 / 复制 / 未归约三种状态可共存）
- **唯一失败情形**：`in/out_shardings` 都空 + `manual_axes` 非空 + **函数体非空**。
  报文：`op has manual_axes when there are no in/out shardings and the body is not empty`
  —— 注意如果**函数体为空**，这种写法是允许的（L1-07 的空体内联）。

## 练习

见第 7 幕。三道题分别考补 replicated、排序依据、失败条件。

## 验收点

- [x] `check_ir_fidelity.py`：34 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- [x] `lint_lessons.py`：语法检查通过
- 自检：给定 manual_axes 与 in/out_shardings，能写出清理后的结果
