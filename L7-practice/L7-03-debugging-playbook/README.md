# L7-03 · `debugging-playbook` — 调试手册（P0）

> 层：**L7 · 综合实战** ｜ 优先级：P0 ｜ 前置课：`L2-07`、`L2-08`、`L2-10`、`L5-09`

## 学习目标

看完这一课，你应该能：

1. **给定一个「坏」IR 能定位原因**（TODOLIST 验收点）；
2. 说出「分片没传播过去」的**第一步检查**是什么；
3. 说出 `--debug-sharding-origins` 与 `--module-dump-directory` 的分工；
4. 说出为什么区域算子最容易出问题。

## 覆盖的测试文件（4 个，全在 `transforms/propagation/test/`）

| 文件 | 行数 | 用例数 | 提供什么线索 |
|---|---|---|---|
| `propagation_pipeline_data_flow_edges.mlir` | 820 | **40** | **区域算子**的传播 |
| `propagation_pipeline_dedup_functions_fully_true.mlir` | 304 | 21 | **选项开关** |
| `op_sharding_rule_registry_conservative.mlir` | 78 | 7 | **保守规则** |
| `op_sharding_rule_registry_failures.mlir` | 8 | 1 | **诊断警告** |

> **本课完成后覆盖率 241/241 = 100%。**

## 场景（6 幕）

1. **★ 四个文件 = 四类调试线索**
2. **★ 最直接的线索：`-verify-diagnostics`**
3. **★ `conservative-propagation`：规则太保守 → 多余通信**
4. **★ `data_flow_edges`：区域算子最容易出问题**
5. **★ 两个调试工具（已实测存在）**
6. **★ 排查清单与三步法**

## 核心结论

### ★ 四类线索

| 症状 | 该看哪个文件 |
|---|---|
| **分片没传播过去** | `data_flow_edges` |
| **出现意外的 all-gather** | `registry_conservative` |
| **算子没有 sharding rule** | `registry_failures` |
| **想切换传播行为** | `dedup_functions_fully_true` |

### ★ 最直接的线索：诊断警告（8 行但最有用）

```mlir
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules -verify-diagnostics

func.func @unknown_custom_op(%arg0: tensor<8x2xui32>, %arg1: tensor<8x2xui32>) -> tensor<8x2xui64> {
  // expected-warning@+1 {{custom call @unknown_custom_op is unknown to SDY sharding rule registry}}
  %0 = stablehlo.custom_call @unknown_custom_op(%arg0, %arg1) : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
```

**当自定义算子「分片没传播过去」时，第一件事**：跑
`sdy_opt -sdy-populate-op-sharding-rules` 看有没有这条警告。

**有警告 → 算子没注册 sharding rule** → 见 **L7-04** 的接入方法。

**关键**：这是 `expected-warning` 而非 `error` —— **缺规则不报错**，只是跳过。

### ★ 保守规则

```mlir
// RUN: sdy_opt %s -sdy-populate-op-sharding-rules="conservative-propagation=true" 2>&1 | FileCheck %s

func.func @concat(%arg0: tensor<4x3x256xf32>, %arg1: tensor<4x5x256xf32>) -> tensor<4x8x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
```

**注意 `k=1, l=1, m=1`** —— 拼接维的因子被设为 **1**（**该维不可分**）。

**后果**：传播不过去 → 插入 reshard → 你看到**意外的 all-gather**。

**排查手法**：切换 `conservative-propagation` 对比 —— 行为变了就说明是规则问题。

### ★ 区域算子最容易出问题

`data_flow_edges` 有 **820 行 / 40 个用例**（本课最大），注释直接点明
`CaseOp` 和 `WhileOp`。

**为什么**：区域算子的数据流要**穿过 region 边界** ——
block argument、`sdy.return`、嵌套结构都可能**打断传播**。

### ★ 两个调试工具（实测确认）

| 工具 | 回答什么问题 |
|---|---|
| `--debug-sharding-origins` | 分片**从哪来**（传播**之前**的标注） |
| `--module-dump-directory=<dir>` | 分片**在哪一步变**（dump 每个 pass 的 IR） |

**实测命令**：`sdy_opt --help` 确认两者都存在
（AGENTS.md §3.1 要求 flag 必须与 `passes.td` 一致）。

### ★ 三步法

| 步 | 做什么 | 工具 |
|---|---|---|
| **①** | **看诊断**（算子有没有规则） | `-verify-diagnostics` |
| **②** | **切选项**（定位到哪个 pass） | `conservative-propagation` / `dedup-functions-fully` |
| **③** | **dump IR 逐步 diff**（精确定位） | `--module-dump-directory` |

**一个重要的心态**：「分片没传播过去」**不报错** —— 这是最难查的一类问题，
因为它是「**静默地跳过**」。所以要**主动检查**诊断信息。

## 验收点

- [x] `check_ir_fidelity.py`：6 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：**241/241 = 100%**，B 组全部通过
- [x] `check_flags.py`：无非法 flag（两个调试 flag 已用 `--help` 实测确认）
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能对给定的「坏」IR 按三步法定位原因
