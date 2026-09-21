# L6-07 · `exec-reshape` — reshape 的执行（P0）

> 层：**L6 · 执行与解释器** ｜ 优先级：P0 ｜ 前置课：`L6-00`、`L4-03`、`L4-09`、`L2-02`

## 学习目标

看完这一课，你应该能：

1. 说出 `reshape` 降级的**三步模式**；
2. 说出为什么需要 `slice`、为什么需要 `reshard`；
3. 说出**子轴**在 reshape 中的用法（`_2groups`）；
4. 说出 `device_ids` 倒序为什么**不影响结果**。

## 覆盖的测试文件（7 个，全在 `executable_partitioner_pipeline/`）

| 族 | 文件 | 行数 |
|---|---|---|
| **子轴** | `stablehlo_reshape_1d_to_2d_split_2groups.mlir` | 69 |
| **pass-through** | `stablehlo_reshape_1d_to_2d_split_passthrough_indivisible.mlir` | 56 |
| **奇数轴** | `stablehlo_reshape_1d_to_2d_split_gap_2.mlir` | 48 |
| **轴移位** | `stablehlo_reshape_axis_shift.mlir` | 48 |
| **基准** | `stablehlo_reshape_1d_to_2d_split.mlir` | 39 |
| **设备号** | `stablehlo_reshape_1d_to_2d_split_custom_device_ids.mlir` | 39 |
| **无关轴** | `stablehlo_reshape_2d_split_unrelated_axis.mlir` | 39 |

**全部 7 个都有两个 RUN 行**（`--enable_halo_exchange=false/true`）
→ 验证 **REPL 与 HALO 等价**（L4-09）。

## 场景（5 幕）

1. **★ `reshape` 降级的三步模式**
2. 基准：一维复合分片拆到两维
3. **★ 7 个文件的分片重排情形**
4. 其余三种：pass-through / 无关轴 / 轴移位
5. 小结

## 核心结论

### ★ 三步模式

```text
%0 = stablehlo.slice %arg0 [...]   // ① 切到可 reshape 的大小
%1 = stablehlo.reshape %0 {...}    // ② reshape 并【重排分片】
%2 = sdy.reshard %1 <...>          // ③ reshard（可能通信）
```

**为什么需要 `slice`**：`reshape` 要求**元素总数不变**，但原形状往往拆不成目标形状
（如 `8` 拆不成 `2x3`）→ 先 `slice` 到能拆的大小（`6`）。

**为什么需要 `reshard`**：`reshape` 后分片可能**不再匹配**（如 `c` 要切 `3/2` 除不尽）
→ 需要通信。

### 基准：`1d_to_2d_split`

```mlir
sdy.mesh @mesh_a_4 = <["b"=2, "c"=2]>

func.func @parallel_reshape_1d_to_2d_split(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}]> : tensor<2x3xi32>
  return %2 : tensor<2x3xi32>
}
```

**核心**：`[{"b","c"}]` → **`[{"b"},{"c"}]`** ——
**一维的复合分片被「拆」到两个维度**。

**逐设备数值**（`iota(8) + 1` = `[1..8]`）：
- `slice [0:6]` → `[1,2,3,4,5,6]`；`reshape` → **`[[1,2,3],[4,5,6]]`**
- 4 台设备各拿 2 个元素；**4 台结果都等于 `%seq`**（因为 reshard 到全复制）

**`iota + 1` 的技巧**：`iota` 从 `0` 开始，第 0 个元素是 `0` ——
可能与「填充值 0」混淆。`+1` 让所有元素**非零**，便于辨识。

### ★ 7 种分片重排情形

| 文件 | 分片变化 | 特点 |
|---|---|---|
| `1d_to_2d_split` | `[{"b","c"}]` → `[{"b"},{"c"}]` | 基准 |
| `_2groups` | `[{"a"},{"b"}]` → 四个**子轴** | **子轴**（L2-02） |
| `_custom_device_ids` | 同基准 + `device_ids=[3,2,1,0]` | **设备号倒序**（L1-02） |
| `_gap_2` | `[{"b","c"}]` → `[{"b"},{"c"}]`，`c=3` | **奇数轴** |
| `_passthrough_indivisible` | `[{"x"},{"b","c"}]` → `[{"x"},{"b"},{"c"}]` | **pass-through** |
| `2d_split_unrelated_axis` | `[{"a"},{"b"}]` → `[{"a"},{},{"b"}]` | **无关轴** |
| `axis_shift` | `[{"a"}]` → `[{},{}]` | **轴移位** |

**`_2groups` 的子轴**：
```mlir
%1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_ab_16, [{"a":(1)2}, {"a":(2)2}, {"b":(1)2}, {"b":(2)2}]>]>} : (tensor<6x14xi32>) -> tensor<2x3x2x7xi32>
```
网格 `a=4, b=4`（16 台设备），**两个维度拆成四个** ——
`a=4` 被拆成两组 `(1)2`/`(2)2`，必须用**子轴**把轴的大小分配下去。

**`_custom_device_ids` 的 `device_ids=[3,2,1,0]`**：
除设备号外与基准**完全相同** → 验证**设备号的映射顺序不影响语义**。

**`2d_split_unrelated_axis`**：`reshape` 到 `1x5x4` ——
**在最前面插入一个大小为 1 的维**，该维**未分片**（因为大小是 1，切不了）。

## 验收点

- [x] `check_ir_fidelity.py`：10 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：7 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出三步模式，以及为什么 `reshard` 有时是必需的
