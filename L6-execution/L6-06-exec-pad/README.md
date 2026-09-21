# L6-06 · `exec-pad` — pad 的执行（P0 · ★ 本层最大族）

> 层：**L6 · 执行与解释器** ｜ 优先级：P0 ｜ 前置课：`L6-00`、`L5-04`、`L5-09`、`L4-09`

## 学习目标

看完这一课，你应该能：

1. **说出 `pad` 为什么是本层最大族**；
2. 说出 **uniform / non-uniform 的判据**；
3. 说出 `pad_non_sharded` 为什么**无通信**；
4. 说出 `indivisible` 文件里 **`high` 从 2 调到 3 的原因**。

## 覆盖的测试文件（19 个）

### `executable_convert_global_to_local/`（3 个 —— 分片语义）

| 文件 | 判据 | 通信 |
|---|---|---|
| `stablehlo_pad_non_sharded.mlir` | padding 在**未分片**维 | **无** |
| `stablehlo_pad_sharded_uniform.mlir` | `pLow + pHigh = pInt` | 无（local 参数相同） |
| `stablehlo_pad_sharded_non_uniform.mlir` | `pLow + pHigh > pInt` | 首尾设备不同 |

### `executable_partitioner_pipeline/`（16 个 —— halo exchange 边界情形）

| 族 | 文件 |
|---|---|
| **跳数** | `single_left_hop`、`single_right_hop`、`multiple_right_hops`、`multiple_hops_right_shift` |
| **方向** | `left_shift`、`right_shift` |
| **大 padding** | `large_pad`、`large_pad_within_one_hop` |
| **多维** | `multidim_mixed_shifts`、`multidim_with_hops` |
| **全复制** | `replicated_dual_slice_pad`、`replicated_negative_high_padding`、`replicated_negative_low_padding` |
| **其它** | `indivisible`、`interior`、`replica_id` |

## 场景（6 幕）

1. **★ 为什么 `pad` 是「本层最大族」**
2. padding 在未分片的维上 → 无通信
3. **★ uniform：`pLow + pHigh = pInt`**
4. **★ non-uniform：`pLow + pHigh > pInt`**
5. **★ indivisible：L5-09 的完整实例**
6. 小结：19 个文件的族谱与三条结论

## 核心结论

### ★ 为什么是最大族

**答案在 16 个 partitioner 文件的命名里**：

| 关键词 | 含义 |
|---|---|
| `hop` | 相邻设备间的数据交换**跳数** |
| `shift` | 数据移动**方向**（`left`/`right`） |
| `large_pad` | padding **跨越多个 hop** |
| `multidim` | **多维**同时 padding |
| `replicated_*` | **全复制**情形 |
| `negative_*_padding` | **负 padding**（裁剪） |

**这些全是 `halo exchange` 的边界情形**（回顾 **L4-09**）：
HALO 需要「和邻居交换边界数据」，而边界情形极多 ——
**跳数 × 方向 × 维度 × 正负 padding** 的每种组合都要验证。

**大部分文件有两个 RUN 行**（`--enable_halo_exchange=true/false`）
→ 验证 **HALO 与 REPL 等价**。

### `pad_non_sharded`：作用维判据

`low = [0, 1], high = [0, 1]` —— **只在第 1 维（未分片维）padding**。

每台设备本地 `2x2` → `2x4`，**local pad 参数相同** → **无通信** ✓

**这是 L5-04 的「作用维判据」在 `pad` 上的体现。**

### ★ uniform / non-uniform

| 情形 | 判据 | 结果 |
|---|---|---|
| **uniform** | `pLow + pHigh = pInt` | 每台 local pad **相同** |
| **non-uniform** | `pLow + pHigh > pInt` | 首/尾设备与中间设备**不同** |

**uniform 的例子**（`low=1, high=0, interior=1`，输入 `[1,2,3,4]`）：

| 设备 | 输入 | 局部结果 |
|---|---|---|
| 0 | `[1, 2]` | **`[0, 1, 0, 2]`** |
| 1 | `[3, 4]` | **`[0, 3, 0, 4]`** |

全局拼接 = `[0, 1, 0, 2, 0, 3, 0, 4]` ✓

**为什么均匀**：`pLow + pHigh = pInt` → 边界 padding 恰好等于 interior padding
→ **段与段的边界看起来和段内一样**。

**non-uniform 的例子**（`low=2, high=2, interior=2`）：

| 设备 | 局部结果 |
|---|---|
| 0 | **`[9, 9, 1, 9]`** |
| 1 | **`[9, 2, 9, 9]`** |

两台设备的 `9` 出现在**不同位置** ✓

**注意填充值是 `9` 而非 `0`** —— 故意用 `9` 以便**区分**填充值与真实数据
（L6-03 讲的「缩放技巧」同源）。

### ★ indivisible：L5-09 的完整实例

```mlir
// The pad input is sliced from 8x2 to 7x2 (indivisible, padded to 8x2).
// The original pad result size (10) is also indivisible by mesh axis size (4).
// PadForDivisibility should adjust high padding of the pad op to 3 (instead of 2)
// to make the result 12x2 (divisible). The result is then trimmed to 10x2
// after the final reshard (all_gather).
```

**六步流程**：
```
7x2（不可整除）
  → pad 到 8x2
  → 结果 10 又不可整除 → high 从 2 调到 3
  → 12x2（12 = 4×3 可整除）
  → all_gather
  → 裁回 10x2 ✓
```

**两个 RUN 行用的是位置参数**（`"true"`/`"false"`）——
正好印证 **L6-00** 讲的脚本参数解析（第 3 个位置参数是 `ENABLE_HALO_EXCHANGE`）。

## 验收点

- [x] `check_ir_fidelity.py`：16 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：19 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 uniform 判据，以及 `indivisible` 里 `high` 为何从 2 调到 3
