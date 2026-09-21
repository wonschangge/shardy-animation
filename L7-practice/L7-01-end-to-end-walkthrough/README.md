# L7-01 · `end-to-end-walkthrough` — 端到端走查（P0 · L7 开篇）

> 层：**L7 · 综合实战** ｜ 优先级：P0 ｜ 前置课：`L6-00`、`L4-10`、`L4-14`、`L1-01`

## 学习目标

看完这一课，你应该能：

1. 说出**空轴网格** `sdy.mesh @m = <[], device_ids=[0]>` 的含义与用途；
2. 说出**网格切换**为什么需要通信；
3. 说出 `collective_permute` 的**完全分解**与**部分分解**的区别；
4. 对着动画**复述完整流程**（TODOLIST 验收点）。

## 覆盖的测试文件（4 个，全在 `executable_partitioner_pipeline/`）

| 文件 | 行数 | 验证什么 | 对应 pass |
|---|---|---|---|
| `sdy_all_to_all_fully_scattered.mlir` | 95 | 通信优化（**完全分解**） | `optimize-collectives`（L4-10） |
| `sdy_all_to_all_partially_scattered.mlir` | 69 | 通信优化（**部分分解**） | 同上 |
| `single_device_switch.mlir` | 52 | **单设备切换** | `resolve-single-device-sharding`（L4-14） |
| `single_device_add.mlir` | 45 | **空轴网格** | 同上 |

## 场景（5 幕）

1. **L7 开篇：完整分区器流水线**
2. **★ 空轴网格：单设备的表达方式**
3. **★ `single_device_switch`：多设备 ↔ 单设备**
4. **★ `collective_permute` 的完全/部分分解**
5. 小结：时间轴上的四个片段

## 核心结论

### 完整分区器流水线（回顾 L6-00）

这四个文件跑的是 `executable_partitioner_pipeline` 的**完整流水线**：

| pass | 对应课 |
|---|---|
| `--sdy-insert-explicit-reshards` | L4-02～07 |
| `--sdy-resolve-permutation-factors` | L4-09 |
| `--sdy-reshard-to-collectives` | L4-08 |
| **`--sdy-optimize-collectives`** | **L4-10** ← 本课两个文件 |
| `--sdy-pad-for-divisibility` | L5-09 |
| **`--sdy-resolve-single-device-sharding`** | **L4-14** ← 本课两个文件 |

### ★ 空轴网格

```mlir
sdy.mesh @mesh = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
```

**`<[], device_ids=[0]>`** —— **轴列表是空的**！一个**只有一个设备、没有任何轴**的网格。

**为什么需要**：单设备情形下**没有「轴」可分**，但仍需一个**合法的 mesh**
来表达「这个张量只在一台设备上」。

**注释很坦诚**：`// It doesn't make much sense to run an element-wise op on single-device in...`
—— 这个测试**本身意义不大**，它存在是为了**覆盖代码路径**。

### ★ 网格切换

```mlir
sdy.mesh @mesh = <["x"=4]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
sdy.mesh @single_dev_2 = <[], device_ids=[2]>
```

同一程序里**多个网格并存** → 从 `x=4` 切到单设备意味着**收拢分片**（需要通信）；
反向切换意味着**广播**。

**`device_ids` 的作用**：`@single_dev_1` 用**物理设备 1** —— **收拢到哪台**是可指定的（L1-02）。

### ★ `collective_permute` 的完全/部分分解

| 行 | `fully` 版本 | `partially` 版本 |
|---|---|---|
| 第 3 句 | `eliminates ... by **decomposing**` | **`detects`** the collective_permute + all_to_all **`chain`** and |
| 第 4 句 | are **`fully`** scattered off dimension 0 | are **`partially`** scattered off dimension 0 (**`only`**) |

- **`fully`**：pass **主动分解** `collective_permute` —— 两轴都散开。
- **`partially`**：pass **检测到**「`collective_permute` + `all_to_all` 链」
  —— **复用已有通信**，只有一部分散开。

**★ 这验证了 L4-10 的「优化模式多样性」**：同一个算子可能被**完全**或**部分**消除。

**优化三步**：① 初次降级产生 `collective_permute` → ② pass 消除它 →
③ 被置换的轴散射到第 0 维之外。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：4 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能复述「完整流水线的六个 pass」与两个被本课验证的 pass
