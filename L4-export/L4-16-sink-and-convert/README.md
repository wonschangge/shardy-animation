# L4-16 · `sink-and-convert` — 边下沉与转换

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L1-08`、`L3-05`、`L1-10`

## 学习目标

看完这一课，你应该能：

1. **预测下沉后分片落在哪个值上**（TODOLIST 验收点）；
2. 说出 `data_flow_edge` 与 `func_data_flow_edge` 的区别；
3. 说出 `sharding_constraint` 为什么要转成 `reshard`；
4. 说出本课在导出流水线中的**位置**（为什么 sink 必须**先跑**）。

## 覆盖的测试文件（3 个 / 468 行 / 30 用例）

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `-sdy-sink-data-flow-edges` | `sink_data_flow_edges.mlir` | 251 | 13 |
| `-sdy-sink-func-data-flow-edges` | `sink_func_data_flow_edges.mlir` | 198 | 15 |
| `-sdy-sharding-constraint-to-reshard` | `sharding_constraint_to_reshard.mlir` | 19 | 2 |

## 场景（6 幕）

1. 共同点：消除「标注型」的 op
2. **★ 边的下沉：分片落到输入值上**
3. 13 个用例的边界与一处测试空缺
4. 函数级边与 `constraint → reshard`
5. 3 个 pass / 30 个用例的族谱
6. 练习

## 核心结论

### 共同点

三个 pass 都在**消除「标注型」的 op** ——
它们都是**传播期的辅助机制**，导出时不再需要：

| op | 定义/插入于 |
|---|---|
| `sdy.data_flow_edge` | L1-08 定义、L3-04 插入 |
| `sdy.func_data_flow_edge` | L3-05 插入、L3-10 搬分片 |
| `sdy.sharding_constraint` | L1-10 定义、传播消费它 |

### ★ 边的下沉

- **边 op 被删除**，后续算子改成**直接读原值**
  （`add %4, %4` → `add %iterArg, %iterArg`）。
- **分片落到边的【输入值】上** —— 具体落点取决于值是谁定义的：

| 边的位置 | 分片落到 |
|---|---|
| **block argument** | `while` 的**操作数**（`%arg0`）上 |
| **算子结果** | **`while` op 本身**上 |

### 两种边的区别

| | `sdy.data_flow_edge` | `sdy.func_data_flow_edge` |
|---|---|---|
| 作用域 | **区域算子内部**（while/case 的 block） | **函数调用**（参数与结果） |
| 覆盖课 | L1-08 / L3-04 | L3-05 / L3-10 |

**共同点**：都是「**桥接**」op —— 传播结束后桥接使命完成 → **下沉并删除**。

### `constraint → reshard`

- **只改 op 名**，分片参数与附加属性（如 `{foo}`）**完全保留**。
- **token** 是无维度的值 → 转成 **0 维 reshard**（`<@mesh, []>`）。

**为什么要转**：
- `sharding_constraint` 是**声明性**的（「我希望」）—— 允许传播器**自由选择**如何满足；
- `reshard` 是**命令性**的（「请搬」）—— 明确的搬运指令。

传播期需要前者（给传播自由度）；传播结束后「我希望」已无意义 → 统一转成 `reshard`。

### 在流水线中的位置

**L4-01 的 NOTE** 说过：
`we apply sdy-add-data-flow-edges first, to make sure sdy-sink-data-flow-edges
is applied before any pass that operated on ShardableDataFlowOpInterface rather
than DataFlowEdgeOp.`

→ **本课的 sink 就是那个「必须先跑」的 pass** ——
因为后续 pass 期望分片**已经落在张量上**，而不是还在边上。

### 一处测试空缺

`sink_data_flow_edges.mlir` 开头有 TODO 注释：
`once ops like while are allowed to have shardings with different meshes,
add a test that verifies that the first mesh name is used for missing shardings.`

→ 「缺 sharding 时用第一个 mesh 名」这条规则**还没有测试覆盖**。
（按 AGENTS.md 的红线：**TODO 描述的是未来计划，不是当前行为**。）

## 练习

见第 6 幕。三道题分别考边的下沉、两种边的区别、约束的转换。

## 验收点

- [x] `check_ir_fidelity.py`：22 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能预测下沉后分片落在哪个值上
