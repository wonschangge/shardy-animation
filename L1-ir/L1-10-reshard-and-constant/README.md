# L1-10 · `reshard-and-constant` — 重分片与常量（L1 收官）

> 层：**L1 · IR 构件** ｜ 优先级：P1 ｜ 前置课：`L1-03`、`L1-06`、`L1-07`

## 学习目标

看完这一课，你应该能：

1. 说出 `sdy.reshard` 的**生命周期**：约束 → 传播 → reshard → collective；
2. 说出 reshard 的两条特有校验（manual 绑定 / 保留轴的归约算子）；
3. 判断一条 reshard 链**能否折叠**（中间结果是否还有别的使用者）；
4. 判断两条 reshard **能否 CSE 合并**（输入与目标分片是否完全一致）；
5. 解释 `sdy.constant` 为什么**故意不实现 ConstantLike、不带 folder**；
6. 回顾 L1 十课的知识地图。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/reshard_verification.mlir` | 66 | 6 个用例（5 报错 + 1 合法反例） |
| `shardy/dialect/sdy/ir/test/reshard_canonicalization.mlir` | 255 | 17 个规范化用例（链式折叠 + CSE） |
| `shardy/dialect/sdy/ir/test/constant_parse_print.mlir` | 15 | 2 个打印用例 |
| `shardy/dialect/sdy/ir/test/constant_verification.mlir` | 15 | 2 条校验错误 |

## 场景（8 幕）

1. `reshard` 的生命周期（三阶段）
2. reshard 的校验：复用 + 两条特有
3. 规范化：链式折叠
4. 规范化：CSE 合并
5. `sdy.constant` 为什么自定义
6. 常量的打印与校验
7. **★ L1 总结：十课知识地图**
8. 练习

## 核心结论

- **生命周期**：`constraint`（用户写）→ 传播消费 → `reshard`（传播插入）→ `collective`（分区器替换）。
  导出后 IR 里**不应再有** `reshard`。
- reshard 的分片校验与普通张量分片**完全复用**；两条特有检查是
  **manual 轴不可用** 与 **保留轴的归约算子不可改**。
- **折叠条件**：中间结果除下一条 reshard 外**没有别的使用者**。
- **CSE 条件**：输入相同且目标分片**完全一致**。
- `sdy.constant` 刻意不带 folder —— 否则贪婪重写器会把常量合并，破坏"每个使用者可独立分片"。
- 常量必须有**静态形状**。

## 练习

见第 8 幕。三道题分别考折叠、CSE、未归约轴规则。

## 验收点

- [x] `check_ir_fidelity.py`：22 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：四个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能判断任意一条 reshard 链能否折叠、能否 CSE
