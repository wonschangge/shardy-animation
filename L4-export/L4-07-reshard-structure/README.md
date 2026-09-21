# L4-07 · `reshard-structure` — 结构性场景的 reshard（L4 算子族收官）

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L1-07`、`L3-04`、`L4-02`

## 学习目标

看完这一课，你应该能：

1. 说出**四类边界**（区域 / 函数 / mesh / 单设备）及各自的 reshard 位置；
2. 说出区域算子的 reshard 为什么常插在**区域内部**；
3. 说出 mesh 切换的判据（轴名 + **设备顺序**）；
4. 说出**四条判据**，能独立判断该插什么 reshard。

## 覆盖的测试文件（9 个 / 1455 行 / 112 用例）

| 文件 | 行数 | 用例数 | 主题 |
|---|---|---|---|
| `insert_explicit_reshards/data_flow_ops.mlir` | 171 | 7 | 区域算子 |
| `insert_explicit_reshards/manual_computation.mlir` | 34 | 2 | 手动计算 |
| `insert_explicit_reshards/call_ops.mlir` | 130 | 11 | 函数调用 |
| `insert_explicit_reshards/call_ops_enable_full_version_false.mlir` | 124 | 11 | 同上（另一选项） |
| `insert_explicit_reshards/func_inputs_outputs.mlir` | 109 | 14 | 函数边界与跨 mesh |
| `insert_explicit_reshards/meshes.mlir` | 183 | 19 | **mesh 切换** |
| `insert_explicit_reshards/single_device_sharding.mlir` | 241 | 10 | 单设备分片 |
| `insert_explicit_reshards/single_device_sharding_errors.mlir` | 30 | 0 | 单设备分片的报错 |
| `insert_explicit_reshards/unreduced.mlir` | 433 | 38 | 未归约轴的完整展开 |

## 场景（6 幕）

1. 四类边界：reshard 就是跨边界的搬运
2. **★ 区域的边界：reshard 插在内部**
3. 函数边界与 mesh 切换
4. 单设备分片与未归约的完整展开
5. **L4 收官：按算子族展开的七课**
6. 练习

## 核心结论

### 四类边界

| 族 | 文件 | 用例数 | 核心问题 |
|---|---|---|---|
| **区域内部** | `data_flow_ops`、`manual_computation` | 9 | reshard 插在区域**内**还是**外** |
| **函数调用** | `call_ops`(×2)、`func_inputs_outputs` | 36 | 实参/形参、return/结果的对齐 |
| **跨 mesh** | `meshes` | 19 | mesh 切换（含**设备序不同**） |
| **单设备** | `single_device_sharding`(×2) | 10 | `maximal` 网格的导出侧 |
| **未归约** | `unreduced` | 38 | L4-02 总纲的完整展开 |

### ★ 区域的边界

- 区域算子（`case` / `while` / `barrier`）内部**可以有 reshard**。
- `case` 的例子：① 分支内算子**之前**（子轴 `"x":(1)2` → 完整轴 `"x"`）；
  ② `stablehlo.return` **之前**（调整到函数结果要求的分片）。
- `manual_computation`：区域**外**满足 `in_shardings`，区域**内**满足 `out_shardings`。
  带 `manual_axes` 时，区域内的 reshard 只在**未冻结**的轴上切换。
- **与 L3-04 的分工**：边负责**跨边界**，reshard 负责**区域内部**。

### mesh 切换

- **判据**（L2-01）：看**轴名与设备顺序**是否一致 —— 名字只是符号。
- `meshes_different_device_order`：轴名与大小都相同、只是**设备顺序不同**，
  仍是**两个网格** → 需要 reshard。
- 跨 mesh 时传播**根本不发生**，只能靠 reshard 显式转换。

### 单设备与未归约

- **单设备分片**：`@maximal_mesh` 的导出侧（L1-01 / L2-01 讲过它的传播规则）；
  五种转换方向；`_errors` 用 `-verify-diagnostics` 锁定**非法组合**。
- **未归约（38 个）**：L4-02 总纲（8 个代表）的**完整版**，用例名高度重叠，可对照阅读。
  新增维度：`all_reduce_delayed_to_call_site` —— 归约能**跨函数**延迟到调用点。

### ★ 四条判据（贯穿 L4-02～07 的 472 个用例）

1. **冲突在哪两侧？** → 决定 reshard 插在**之前**还是**之后**
2. **有没有 `reduction` 因子？** → 有则 `unreduced` + `all_reduce`
3. **结果能不能靠归约合并？** → 不能（顺序、随机序列）则**全复制**
4. **是不是跨边界？** → 区域 / 函数 / mesh / 单设备 → 必须显式转换

## 练习

见第 6 幕。三道题分别考区域内部、mesh 切换、四条判据。

## 验收点

- [x] `check_ir_fidelity.py`：8 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：9 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出四类边界各自该在哪里插 reshard
