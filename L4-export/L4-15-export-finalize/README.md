# L4-15 · `export-finalize` — 导出收尾（P0）

> 层：**L4 · 导出流水线** ｜ 优先级：P0 ｜ 前置课：`L2-01`、`L3-07`、`L4-08`

## 学习目标

看完这一课，你应该能：

1. **给一段 IR，说出该跑哪几个收尾 pass、顺序如何**（TODOLIST 验收点）；
2. 说出不可整除的输入输出分片**怎么被截断**；
3. 说出**为什么要移除输入输出上的子轴**；
4. 说出 `inline_meshes` 与 L3-07「提升」的**镜像关系**。

## 覆盖的测试文件（7 个 / 593 行 / 58 用例）

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `-sdy-close-shardings` | `close_shardings.mlir` | 92 | 15 |
| `-sdy-update-non-divisible-input-output-shardings` | `update_non_divisible_input_output_shardings.mlir` | 182 | 21 |
| `-sdy-remove-sub-axes-in-input-output-shardings` | `remove_sub_axes_in_input_output_shardings.mlir` | 108 | 6 |
| `-sdy-drop-sharding-rules` | `drop_sharding_rules.mlir` | 35 | 4 |
| `-sdy-remove-sharding-groups` | `remove_sharding_groups.mlir` | 12 | 1 |
| `-sdy-inline-meshes` | `inline_meshes.mlir` | 138 | 9 |
| `-sdy-drop-sharding-and-mesh` | `drop_sharding_and_mesh.mlir` | 26 | 2 |

**共同点**：这 7 个 pass **全都在「删东西」** —— 目标是给后端一个**干净且自包含**的 IR。

## 场景（6 幕）

1. 收尾阶段：全都在「删东西」
2. **★ 分片的收尾：闭合、截断、只动边界**
3. 删辅助信息：`sharding_rule` 与 `sharding_group`
4. **★ 网格处理与收尾顺序**
5. 7 个 pass 的族谱与小结
6. 练习

## 核心结论

### ① 分片本身（三个 pass）

- **`close_shardings`**：把**开维 `?`** 闭合 ——
  `{"x", ?}` → `{"x"}`、`{?}` → `{}`。
  传播期保留 `?` 便于调整，**导出时必须定下来**。
- **`update_non_divisible_input_output_shardings`**（★ 最实用）：
  | 情形 | 处理 |
  |---|---|
  | **部分整除**（`tensor<2>` vs `x=4`） | **子轴截断** `{"x":(1)2}`（只用前 2 个设备） |
  | **完全不能整除**（`tensor<3>` vs `y=2`） | **整个轴去掉** `{}` |
- **`remove_sub_axes_in_input_output_shardings`**：
  **只动输入输出，不动中间张量**。
  移除子轴及其**前面**的轴（只保留最粗的一段）。

### ② 辅助信息（两个 pass）

- **`drop_sharding_rules`**：删 `sdy.sharding_rule`（**保留** `sdy.sharding`）——
  规则只在**传播期**用（L2-01）。
- **`remove_sharding_groups`**：删 `sdy.sharding_group` ——
  它是「强制同分片」的**承诺**（L3-09），传播结束就**已兑现**。

### ③ 网格（两个 pass，**二选一**）

- **`inline_meshes`**：命名网格 → 内联（后端要**自包含**的 IR）。
  **与 L3-07 的「提升」正好相反**。
- **`drop_sharding_and_mesh`**：全部删掉 → 变回**纯 StableHLO**
  （后端完全不用 Shardy 时）。

### ★ 收尾顺序（验收点答案）

| 阶段 | pass | 为什么在这个位置 |
|---|---|---|
| **① 分片本身** | `close_shardings`、`update_non_divisible_...`、`remove_sub_axes_...` | 后续步骤都依赖「分片已确定」 |
| **② 辅助信息** | `drop_sharding_rules`、`remove_sharding_groups` | 只在传播中有用 |
| **③ 网格** | `inline_meshes`、`drop_sharding_and_mesh` | 分片**引用**网格 —— 必须先定分片 |

**一句话**：**先定分片、再删辅助信息、最后处理网格**。

## 练习

见第 6 幕。三道题分别考不可整除、只动边界、收尾顺序。

## 验收点

- [x] `check_ir_fidelity.py`：29 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：7 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给一段 IR，能说出该跑哪几个收尾 pass、顺序如何
