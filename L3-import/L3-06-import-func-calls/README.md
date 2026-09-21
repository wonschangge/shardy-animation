# L3-06 · `import-func-calls` — 内联函数调用

> 层：**L3 · 导入流水线** ｜ 优先级：P2 ｜ 前置课：`L1-08`、`L3-05`

## 学习目标

看完这一课，你应该能：

1. 说出 `call` → `named_computation` 时同时发生的**四件事**；
2. 说出 **`in_shardings` 与 `out_shardings` 的来源不同**；
3. 说出属性搬运为什么必须做；
4. 说出 `add-data-flow-edges-on-named-computations=false` 解决什么问题；
5. 说出 `flatten_call_graph` 的用途与难点。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `shardy/dialect/sdy/transforms/import/test/import_func_calls.mlir` | 634 | 21 | `-sdy-import-func-calls` |
| `.../import_func_calls_add_data_flow_edges_on_named_computations_false.mlir` | 572 | 21 | 同上 + `add-data-flow-edges-on-named-computations=false` |
| `shardy/dialect/sdy/transforms/import/test/flatten_call_graph.mlir` | 415 | 13 | `-sdy-flatten-call-graph` |

## 场景（6 幕）

1. 核心变换：`call` → `named_computation`
2. **★ `in_shardings` 与 `out_shardings` 的来源不同**
3. 属性搬运与多个调用点
4. 一个开关与一个配套 pass
5. 21 个用例的族谱
6. 练习

## 核心结论

- **四件事同时发生**：
  ① 换名字（取自**被调函数名**）
  ② 搬函数体（区域内按 L3-04 的规则插边）
  ③ 搬属性（`backend_config` / `inlineable` / 自定义，原样保留）
  ④ **删除原私有函数**
- **两个来源不同**（本课最容易搞错的一点）：
  | 字段 | 来源 |
  |---|---|
  | `in_shardings` | **被调函数**的参数标注 |
  | `out_shardings` | **调用点**的 `sdy.sharding` |
  函数结果上的标注**被忽略**（测试 NOTE：`we ignore any arg/result shardings on the function`）。
  **为什么以调用点为准**：同一函数可能被多个调用点调用，各调用点可要求不同分片。
- **属性搬运必须做**：`backend_config` 等会影响后端行为，丢了就改变语义。
- **多个调用点**各生成一个 `named_computation`（**同名**允许 —— 名字只用于可读性与 outline）。
- **`add-data-flow-edges-on-named-computations=false`**：关掉自动插边，避免与
  L3-04 的 `-sdy-add-data-flow-edges` **重复工作**。
- **`flatten_call_graph`**：处理调用图**结构**；难点在于**分片不匹配**时怎么办
  （3 个用例专门覆盖）。

## 用例族谱（21 个）

| 族 | 数量 | 族 | 数量 |
|---|---|---|---|
| 属性搬运 | 5 | 多参数分片 | 6 |
| 多调用点 | 2 | 结果分片对比 | 5 |
| 非扁平调用图 | 2 | 命名 | 1 |

## 练习

见第 6 幕。三道题分别考四件事、两个来源、开关的作用。

## 验收点

- [x] `check_ir_fidelity.py`：23 个 IR 块全部逐字来自源文件
- [x] `check_coverage.py`：三个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定一个 call，能写出变换后的 named_computation 骨架
