# L6-05 · `exec-gather` — gather 的执行（P0 · ★ 最复杂）

> 层：**L6 · 执行与解释器** ｜ 优先级：P0 ｜ 前置课：`L6-00`、`L5-08`、`L5-09`

## 学习目标

看完这一课，你应该能：

1. 说出 gather 的索引重映射**在数值上如何被验证**；
2. **说出 `sum`/`min`/`max` 三种归约的填充值分别是什么**；
3. 说出为什么 `min`/`max` **必须单独测试**；
4. 说出「先 diff 变体文件」这个读代码方法的价值。

## 覆盖的测试文件（6 个）

| 文件 | 验证什么 | 与基准的差异 |
|---|---|---|
| `.../stablehlo_gather_shard_reduction_dim_is_collapsed.mlir` | 基准（`sum` 归约） | — |
| `.../stablehlo_gather_shard_reduction_dim_is_collapsed_i32.mlir` | **索引类型无关** | `i64` → `i32` |
| `.../stablehlo_gather_shard_reduction_dim_is_collapsed_min.mlir` | **`min` 归约** | 填充值 `0` → **`+∞`** |
| `.../stablehlo_gather_shard_reduction_dim_is_collapsed_max.mlir` | **`max` 归约** | 填充值 `0` → **`−∞`** |
| `.../stablehlo_gather_shard_reduction_dim_is_collapsed_not_in_start_index_map.mlir` | collapsed 维不在 map | 结果 `2x2` → `2x1` |
| `.../stablehlo_gather_shard_two_reduction_dims.mlir` | **两个归约维** | 网格 `2x2`、`unreduced={"x","y"}` |

网格：`sdy.mesh @mesh_2 = <["x"=2]>`

## 场景（6 幕）

1. **★ 六个文件，验证 L5-08 的索引重映射**
2. 基准场景：索引跨两台设备
3. **★★ 实测发现：填充值是归约的单位元**
4. **★ 一个跨课闭环：L5-08 → L5-09 → L6-05**
5. 其余三个变体
6. 小结：最复杂一课的三个收获

## 核心结论

### 基准场景：索引跨两台设备

规则：`([i, j], [k]) -> ([k, j]) reduction={i}`

`%indices = [1, 3]` **是共享的**（两台设备拿同一份），但：

| 设备 | 持有的行 | 索引 `1` | 索引 `3` |
|---|---|---|---|
| 0 | 第 0~1 行 | **属于我** → `[3,4]` | 不属于 → 填填充值 |
| 1 | 第 2~3 行 | 不属于 → 填填充值 | **属于我** → `[7,8]` |

`all_reduce` 合并 → 两台都得到 `[[3,4],[7,8]]` = `%seq` ✓

**这验证了 L5-08 八步中的第 ⑤⑦⑧ 步**（mask / select 填充 / `all_reduce`）。

### ★★ 实测发现：填充值是归约的单位元

用 `sdy_opt` 复现三个变体的降级输出（命令与 `run_sdy_interpreter_test.sh` 一致），
**只有两处不同**：

| 归约种类 | `select` 的填充值 | 十六进制 | 含义 |
|---|---|---|---|
| **`sum`**（基准） | `dense<0.000000e+00>` | `0x00000000` | **`0`** |
| **`min`** | `dense<0x7F800000>` | `0x7F800000` | **`+∞`** |
| **`max`** | `dense<0xFF800000>` | `0xFF800000` | **`−∞`** |

**★ 填充值是该归约的「单位元」（identity element）**：
- `sum` → **`0`**（`0 + x = x`）
- `min` → **`+∞`**（`min(+∞, x) = x`）
- `max` → **`−∞`**（`max(−∞, x) = x`）

**★ 为什么 `min`/`max` 必须单独测试**：
用 `sum` 的 `0` 去配 `min` 归约会**错** —— 因为 `min(0, 3) = 0` 而不是 `3`。

### ★ 跨课闭环

| 课 | 当时说的 | 本课的实测 |
|---|---|---|
| **L5-08** | gather 的 mask「**填 0**」 | **只对 `sum` 成立** —— 那是基准文件的情形 |
| **L5-09** | 「填充值必须是该运算的**单位元**」 | **完全证实** —— gather 也遵守这条 |

**所以 L5-08 的表述需要深化**：「填零」是 `sum` 归约的**特例**，
通用表述应该是「填该归约的**单位元**」。

**这不是 L5-08 讲错了** —— 而是当时只看了基准文件。
**跑遍三个变体才看到完整规律** —— 这正是 AGENTS.md §3.3
「凡测试文件里写着但没见过实际输出的，必须跑一遍确认」那条红线的价值。

### 一个可推广的读代码方法

**看到多个变体文件时，先 `diff`** —— 立刻看出「变化的维度是什么」：

| diff 结果 | 结论 |
|---|---|
| `_i32` vs 基准 | 只差索引类型 → **不影响降级逻辑** |
| `_min`/`_max` vs 基准 | 差**填充值**与 reducer 算子 |
| `not_in_start_index_map` vs 基准 | 差**结果形状**（`2x2` → `2x1`） |

## 验收点

- [x] `check_ir_fidelity.py`：8 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：6 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出三种归约的填充值，并解释为什么 `min` 不能填 `0`
