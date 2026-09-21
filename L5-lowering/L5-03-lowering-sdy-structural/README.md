# L5-03 · `lowering-sdy-structural` — 结构算子降级

> 层：**L5 · 传播与降级** ｜ 优先级：P1 ｜ 前置课：`L1-07`、`L1-08`、`L5-01`

## 学习目标

看完这一课，你应该能：

1. **预测 `manual_computation` 展开后的形状**（TODOLIST 验收点）；
2. 说出 `sdy.constant` 的 **splat 与 dense 为什么处理方式不同**；
3. 说出 `named_computation` 为什么被**保留**；
4. 用一句话概括三个结构算子的降级方式。

## 覆盖的测试文件（3 个 / 314 行 / 14 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/sdy_manual_computation.mlir` | 208 | 8 |
| `convert_global_to_local/sdy_named_computation.mlir` | 58 | 2 |
| `convert_global_to_local/sdy_constant.mlir` | 48 | 4 |

## 场景（5 幕）

1. 三个结构算子，三种降级方式
2. **★ `sdy.constant`：splat 不需要切片**
3. `named_computation`：保留，只改类型
4. **★ `manual_computation`：直接展开**
5. 三个结构算子的对照与小结

## 核心结论

### 三种降级方式

| 结构算子 | 降级方式 | 为什么 |
|---|---|---|
| `sdy.constant` | **按分片裁剪** | 内容需要「切」 |
| `sdy.named_computation` | **保留**，改类型 | 留给 L4-12 outline |
| `sdy.manual_computation` | **展开**（区域消失） | 区域内外的分片状态相同 |

> **注意**：TODOLIST 说 `named_computation` 是「内联」，但**实际测试显示它被保留了**
> —— 以实际 IR 为准。

### ★ `sdy.constant` 的三种情形

| 情形 | 常量类型 | 分片 | 处理 |
|---|---|---|---|
| ① | **splat** | 有 | 直接变**局部形状**的常量（**不需要切片**） |
| ② | **dense** | 有 | 全局常量 + `partition_id` + 切片（六步） |
| ③④ | 任意 | 无 | 直接转换，**类型不变** |

**核心洞察**：**splat 常量不需要切片** —— 所有元素相同，"切出来"和"原样"没区别。
这解释了为什么有些分片常量降级后很「重」、有些却很「轻」。

### `named_computation`

- **op 被保留**，只有**类型**变成局部（函数签名 / block arg / 操作数 / `sdy.return`）。
- **`in_shardings` / `out_shardings` 不变** —— 它们记录「怎么切的」。
- 嵌套也**递归处理**。
- 结构留给 **L4-12** 的 outline 处理。

### ★ `manual_computation` 展开

- **区域完全消失**，内部算子**内联到外层**。
- 例：`abs` → `add` → `tanh` → `return`，**四行线性代码**。
- **为什么可以展开**：`manual_axes` 表示「区域内这些轴不再分片」，
  而导出后**所有张量已经是局部的** —— 「区域内不分片」和「外层」**是同一回事**。

**验收点答案**：展开后的形状 = 区域内的**局部形状** —— 区域内外一致。

**8 个用例**：无自由轴 / **有自由轴**（展开时保留其分片）/ 嵌套 /
**区域内含 5 种集合通信**。

## 验收点

- [x] `check_ir_fidelity.py`：27 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能预测 `manual_computation` 展开后的形状
