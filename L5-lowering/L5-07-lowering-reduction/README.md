# L5-07 · `lowering-reduction` — 归约降级

> 层：**L5 · 传播与降级** ｜ 优先级：P1 ｜ 前置课：`L4-05`、`L5-02`、`L5-05`

## 学习目标

看完这一课，你应该能：

1. 说出归约类降级的**一条判据**（归约是否跨设备）；
2. 说出**多结果归约为什么不能用 `all_reduce`**、该怎么处理；
3. 说出 `reduce_window` 需要通信的**唯一**情形；
4. 说出"单结果 vs 多结果"两种合并方式的选择依据。

## 覆盖的测试文件（2 个 / 148 行 / 5 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_reduce.mlir` | 88 | 3 |
| `convert_global_to_local/stablehlo_reduce_window.mlir` | 60 | 2 |

## 场景（5 幕）

1. **★ 五种情形：归约是否跨设备**
2. 基础情形与 `all_reduce`
3. **★ 多结果归约：`all_gather` + 重新 reduce**
4. `reduce_window`：窗口是否跨设备
5. 五种情形的对照与小结

## 核心结论

### 五种情形

| 文件 | 用例 | 分片位置 | 通信 |
|---|---|---|---|
| `reduce` | `stablehlo_reduce` | 无 | 无 |
| `reduce` | `..._fallback_all_reduce` | **归约维** | **`all_reduce`** |
| `reduce` | `..._multi_result_sharded` | 归约维 + 多结果 | **`all_gather` + reduce** |
| `reduce_window` | `batch_sharded` | 批维（窗口=1） | 无 |
| `reduce_window` | `..._stride_greater_than_window` | 窗口维（**步长≥窗口**） | **无** |

### ★ 一条判据

> **归约类算子的通信需求 = 「归约是否跨设备」。**

- `reduce`：归约维被分片 → 跨设备 → 需要通信
- `reduce_window`：窗口**跨设备边界** → 需要通信；否则不需要

### ★ 两种合并方式

| 情形 | 方式 | 为什么 |
|---|---|---|
| **单结果** | `all_reduce` | 归约**可结合**，直接合并部分结果 |
| **多结果** | **`all_gather` + 重新 `reduce`** | 结果间**有依赖**，不能分别合并 |

**多结果为什么不能用 `all_reduce`**：reducer 是 **argmax 模式** ——
同时算出最大值与位置，两者**互相依赖**。分别 `all_reduce` 会导致
「最大值来自哪个设备」不一致 → 结果错误。

**解法（五步）**：局部 `reduce` → `reshape` → **`all_gather`**（收齐数据）→
**重新 `reduce`**（在完整数据上）→ 返回。

### `reduce_window` 的巧妙用例

`reduce_window_stride_greater_than_window`：**分片维上确实有窗口**
（`window_dimensions = [1, 2]`），但因为 **`window_strides = [1, 2]`**
（步长 = 窗口），每个窗口恰好落在**一台设备的分片内** → **仍无通信**。

### 基础用例的一个细节

`stablehlo_reduce` 的 reducer 内部有**两个 `add`**（非标准的归约函数）——
这验证了转换器**不分析 reducer 语义**，只做类型/分片层面的处理。

## 验收点

- [x] `check_ir_fidelity.py`：68 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定归约算子的分片位置，能说出是否需要通信、用什么方式
