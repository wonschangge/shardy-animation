# L1-09 · `constraint-group-barrier` — 约束类算子

> 层：**L1 · IR 构件** ｜ 优先级：P0 ｜ 前置课：`L1-03`、`L1-07`

## 学习目标

看完这一课，你应该能：

1. 说出三个算子的分工：**钉分片**（constraint）、**强制同分片**（group）、**截断方向**（barrier）；
2. 区分 `sharding_constraint` 的**悬空**与**有使用者**两种语义；
3. 说出 constraint 的三条特殊校验（manual 绑定 / 归约算子 / token）；
4. 说出 `propagation_barrier` 三种方向的含义，以及为什么 `BOTH` 被拒绝；
5. **验证测试文件里的一句话**：区分 `// CHECK:`（指令）与 `// CHECK`（注释）。

## 覆盖的测试文件

| 文件 | 行数 | 覆盖方式 |
|---|---|---|
| `shardy/dialect/sdy/ir/test/sharding_constraint_verification.mlir` | 60 | 5 条校验错误（含 2 条 manual 绑定冲突） |
| `shardy/dialect/sdy/ir/test/sharding_group_parse_print.mlir` | 8 | `group_id` 语法 + **一处过时注释** |
| `shardy/dialect/sdy/ir/test/propagation_barrier_parse_print.mlir` | 32 | 三个方向 + 带分片的屏障 |
| `shardy/dialect/sdy/ir/test/propagation_barrier_verification.mlir` | 7 | `BOTH` 被拒绝 |

## 场景（7 幕）

1. 三个「影响传播」的算子
2. `sharding_constraint`：悬空 vs 有使用者
3. constraint 的三条特殊校验
4. `sharding_group`：强制一组同分片
5. `propagation_barrier`：截断方向
6. **★ 陷阱：测试文件里的注释不一定是真的**
7. 练习

## 核心结论

- **悬空 vs 有使用者**是 `sharding_constraint` 最容易搞错的点：
  悬空 → 约束输入张量本身；有使用者 → 只约束这些使用者。
- constraint 的校验大部分**复用** L1-02/L1-03 的通用规则；
  唯一独有的一条是：**不能改已有未归约轴的归约算子**。
- `sdy.sharding_group` 语法极简（`$input group_id=N`，无结果），语义是组内「一荣俱荣」。
- `propagation_barrier` 允许 `FORWARD` / `BACKWARD` / `NONE`；`BOTH` 被拒绝（等于没有屏障）。
- **`// CHECK:` 是指令，`// CHECK` 只是注释**。测试文件里可能有引用已删除属性的过时注释。

### ⚠ 本课发现的上游问题

`sharding_group_parse_print.mlir` 第 5 行声称输出会含 `type=AS`，但：

| 验证 | 结果 |
|---|---|
| 原样跑 `sdy_opt … \| FileCheck …` | exit **0**（通过） |
| 把该行补上冒号变成真指令后再跑 | exit **1**，`expected string not found` |
| `ops.td` 中的 `Sdy_ShardingGroupOp` 定义 | `arguments = (ins AnyRankedTensor:$input, I64Attr:$group_id)` —— **无 `type` 属性** |

结论：那行**缺冒号，不是生效指令**，`type=AS` 属性不存在，属过时残留。

## 练习

见第 7 幕。三道题分别考算子选择、constraint 语义、屏障方向。

## 验收点

- [x] `check_ir_fidelity.py`：13 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：四个文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 `// CHECK` 与 `// CHECK:` 的区别，并知道如何验证
