# L6-03 · `exec-convolution` — 卷积的执行

> 层：**L6 · 执行与解释器** ｜ 优先级：P1 ｜ 前置课：`L6-00`、`L5-06`、`L4-09`

## 学习目标

看完这一课，你应该能：

1. 说出四个 `shard_*` 文件分别验证哪种分片维度；
2. 说出**三个测试设计技巧**及其目的；
3. 说出 `@sequential_conv` 的**两种来源**；
4. 说出 `dual_semantics` 文件的**两个 RUN 行**验证什么。

## 覆盖的测试文件（5 个）

| 文件 | 分片的维 | `group_count` |
|---|---|---|
| `executable_convert_global_to_local/stablehlo_convolution_shard_batch.mlir` | 批维 `b` | 1 |
| `executable_convert_global_to_local/stablehlo_convolution_shard_batch_group.mlir` | 批维 + 权重输出维 | **2** |
| `executable_convert_global_to_local/stablehlo_convolution_shard_contracting_dim.mlir` | **输入通道 `i`**（收缩维） | 1 |
| `executable_convert_global_to_local/stablehlo_convolution_shard_feature_group.mlir` | 特征维 `f` | **2** |
| `executable_partitioner_pipeline/stablehlo_convolution_dual_semantics_factor_indivisible.mlir` | （验证 REPL vs HALO） | 1 |

网格：`sdy.mesh @mesh_2 = <["x"=2]>`

## 场景（6 幕）

1. **★ 四个文件 = 四种分片维度**
2. **★ 技巧一：用 `@sequential_conv` 现算期望值**
3. **★ 技巧二：手动切分输入**
4. **★ 技巧三：缩放让贡献可辨识**
5. `shard_contracting_dim`：验证 L5-06 的核心规律
6. 小结：五个文件与三个技巧

## 核心结论

### 与 L5-06 的对应

| 文件 | 验证什么 | 对应课 |
|---|---|---|
| `shard_batch` | 批维分片（**无通信**） | L5-06 |
| `shard_batch_group` | `batch_group_count` 改变批维 | L5-06 |
| `shard_contracting_dim` | 收缩维分片需 **`all_reduce`** | L5-06 |
| `shard_feature_group` | `feature_group_count` + 特征维分片 | L5-06 |
| `dual_semantics_factor_indivisible` | **REPL vs HALO 等价** | **L4-09** |

**两课分工**：L5-06 讲 **IR 形态**，L6-03 验证 **数值正确性**。

### ★ 三个测试设计技巧

**① 用 `@sequential_conv` 现算期望值**

```mlir
%expected = func.call @sequential_conv(%lhs, %rhs) : (...) -> ...
```

- **卷积的期望值很难手算**（窗口 + 步长 + padding 的组合）→ 直接调串行版算。
- `@sequential_conv` 的**两种来源**：
  - **脚本自动生成**（若 part1 里只有 `@parallel_conv`）—— L6-00 讲过
  - **手写**（若 part1 里显式写了；脚本的 `if` 条件不会覆盖）

**对比**：L6-01 用硬编码常量、L6-02 用 `concatenate` 现算、本课用**串行版现算**。

**② 手动切分输入**

`slice` 出 `%l0`/`%l1` 分别喂给两台设备，`concatenate` 拼回后与串行结果比较
—— 手工模拟「**分片 → 各算各的 → 拼接**」的全过程。

**③ 缩放让贡献可辨识**

```mlir
// Scale Shard 1 by 1000 to make its contribution identifiable in the sum
%c1000 = stablehlo.constant dense<1000> : tensor<2x4x4x2xi32>
%l1 = stablehlo.multiply %l1_unscaled, %c1000 : tensor<2x4x4x2xi32>
```

**为什么需要**：两个 shard 数值**相近**时，「正确累加」与「只算一个」的结果
可能**碰巧接近** → 测试区分不出来。缩放后贡献差异被**放大** → **敏感度提高**。

**这是通用技巧**：验证「多部分贡献都被合并」时，**给各部分不同的量级**。
Shardy 测试普遍用 `1/10/100/1000` 这种悬殊量级（L6-01/L6-02 也是）。

### `dual_semantics` 的两个 RUN 行

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false
```

验证 **L4-09 的 `enable-halo-exchange` 选项** → 证明
**REPL（全复制）与 HALO（halo exchange）语义等价**。

**两个细节**：它是 `f32`（其他 4 个是 `i32`）；它在**另一个目录**
（`executable_partitioner_pipeline/`），跑的是**完整分区器流水线**。

## 验收点

- [x] `check_ir_fidelity.py`：16 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：5 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出三个测试设计技巧，以及 `@sequential_conv` 的两种来源
