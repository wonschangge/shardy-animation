# L4-02 · `reshard-insertion-overview` — reshard 插入总纲

> 层：**L4 · 导出流水线** ｜ 优先级：P0 ★ 本节总纲 ｜ 前置课：`L1-02`、`L4-01`

## 学习目标

看完这一课，你应该能：

1. 说出**为什么传播后还需要插入 reshard**（本课中心思想）；
2. 说出 `unreduced` 的**延迟归约**策略与"完全不插"的条件；
3. 说出 `dot` 的 reshard 规则（对应维同分片 + 冲突时选一侧）；
4. 说出四个用例族的分工。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `shardy/dialect/sdy/transforms/export/test/insert_explicit_reshards.mlir` | 682 | 53 |
| `shardy/dialect/sdy/transforms/export/test/insert_func_call_reshards.mlir` | 488 | 42 |

RUN 行：
```
-allow-unregistered-dialect -sdy-insert-explicit-reshards='enable-full-version=false mark-partial-result-with-unreduced-axes=true'
-sdy-insert-func-call-reshards
```

## 场景（6 幕）

1. 中心思想：传播定分片，reshard 对齐冲突
2. **★ 最精彩的一族：尽量延迟归约**
3. **★ 最大的一族：`dot` 的 reshard**
4. 其余三族：拼接、状态转换、函数边界
5. 95 个用例的族谱
6. 练习

## 核心结论

- **中心思想**：传播是**各方向独立推导**的，可能出现算子约束不满足
  （如 `dot` 的收缩维在两操作数上不同分片）。
  **传播决定"每个张量怎么切"，reshard 负责"把不兼容的地方对齐"。**
- **★ 延迟归约**（8 个用例）：`unreduced` 轴上的 `all_reduce` **越晚越好** ——
  未归约时每个设备只算自己那一份部分和。
  五种时机：立刻 / 延迟到某算子前 / 部分延迟 / 完全延迟到 return / **完全不插**
  （结果也声明 `unreduced` 时，归约责任交给调用者）。
- **★ `dot` 的 reshard**（24 个用例，最大一族）：
  **算子自身要满足"对应维同分片"；冲突时选一侧，另一侧用 reshard 补齐。**
  例：lhs 第 0 维 `{"y"}` vs 结果 `{"x"}` → dot 采用 `[{"y"}, {}]`，
  再 `reshard` 到 `[{"x"}, {}]`。
  **注意**：收缩维冲突的用例**没有**插 reshard —— 并非所有冲突都这样解决。
- **其余三族**：
  - `concatenate`（3）：拼接维上各操作数必须同分片
  - 状态转换（17）：分片 / 复制 / 未归约的互转（L1-02 的三种正交状态）
  - 函数边界（42）：`-sdy-insert-func-call-reshards` 处理调用点与函数体的对齐
    （**桥接不等于对齐** —— L3-05 的边只做桥接）

## 练习

见第 6 幕。三道题分别考中心思想、延迟归约、dot 的规则。

## 验收点

- [x] `check_ir_fidelity.py`：19 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：两个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过（本课被它抓到 4 处笔误）
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一个 dot 冲突，能说出该插在哪一侧、reshard 放哪里
