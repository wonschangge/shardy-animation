<!-- sdy-coverage
transforms/export/test/executable_partitioner_pipeline/single_device_add.mlir
transforms/export/test/executable_partitioner_pipeline/single_device_switch.mlir
transforms/export/test/executable_partitioner_pipeline/sdy_all_to_all_fully_scattered.mlir
transforms/export/test/executable_partitioner_pipeline/sdy_all_to_all_partially_scattered.mlir
-->

# L7-01 · end-to-end-walkthrough — 源 IR

**L7 综合实战开篇**：四个**端到端流水线**测试 ——
它们跑的是 `executable_partitioner_pipeline` 的**完整分区器流水线**（**L6-00** 讲的第二个脚本）。

| 文件 | 行数 | 验证什么 |
|---|---|---|
| `executable_partitioner_pipeline/sdy_all_to_all_fully_scattered.mlir` | 95 | **通信优化**（全散射） |
| `executable_partitioner_pipeline/sdy_all_to_all_partially_scattered.mlir` | 69 | **通信优化**（部分散射） |
| `executable_partitioner_pipeline/single_device_switch.mlir` | 52 | **单设备切换** |
| `executable_partitioner_pipeline/single_device_add.mlir` | 45 | **单设备加法** |

---

## 一、★ 完整分区器流水线（回顾 L6-00）

这四个文件都用 `run_sdy_interpreter_test.sh` 驱动，而
`executable_partitioner_pipeline/` 版本的脚本跑的是**完整流水线**：

| pass | 对应课 |
|---|---|
| `--sdy-insert-explicit-reshards` | **L4-02～07** |
| `--sdy-resolve-permutation-factors` | **L4-09** |
| `--sdy-reshard-to-collectives` | **L4-08** |
| `--sdy-optimize-collectives` | **L4-10** |
| `--sdy-pad-for-divisibility` | **L5-09** |
| `--sdy-resolve-single-device-sharding` | **L4-14** |

**★ 本课的四个文件正好覆盖其中两个 pass**：
- `single_device_*` → **`--sdy-resolve-single-device-sharding`**（L4-14）
- `sdy_all_to_all_*` → **`--sdy-optimize-collectives`**（L4-10）

---

## 二、`single_device_add`：**空轴网格**

```mlir
// No need to run this test without HALO export.
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// It doesn't make much sense to run an element-wise op on single-device in
```

```mlir
sdy.mesh @mesh = <["x"=2]>
```

```mlir
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
```

```mlir
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
```

**读法**（**本课最特别的一处**）：
- **`sdy.mesh @single_dev_0 = <[], device_ids=[0]>`** ——
  **空轴列表 `[]`**！这是一个**只有一个设备、没有任何轴**的网格。
- `device_ids=[0]` —— 它映射到**物理设备 0**。
- 同文件还定义了 `@mesh`（`x=2`，两台设备）与 `@single_dev_1`（物理设备 1）。

**★ 为什么需要"空轴网格"**：
> **单设备情形下没有"轴"可分** —— 但仍然需要一个**合法的 mesh** 来表达
> "这个张量只在一台设备上"。

**注释里的两句话**：
1. `// No need to run this test without HALO export.` ——
   只用 `"false"`（不用 HALO）跑，因为**单设备不需要 halo exchange**。
2. `// It doesn't make much sense to run an element-wise op on single-device in` ——
   承认这个测试**本身意义不大**（逐元素算子在单设备上没什么可分的），
   它存在是为了**覆盖这条代码路径**。

**★ 这正是 **L4-14** 讲的 `resolve-single-device-sharding` 的场景** ——
把"单设备分片"解析成**不需要通信**的形式。

**注意 RUN 行用位置参数 `"false"`** ——
第 3 个位置参数就是 `ENABLE_HALO_EXCHANGE`（**L6-00** 讲的）。

---

## 三、`single_device_switch`：从多设备**切换到**单设备

```mlir
// No need to run this test without HALO export.
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
sdy.mesh @mesh = <["x"=4]>
```

```mlir
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
```

```mlir
sdy.mesh @single_dev_2 = <[], device_ids=[2]>
```

```mlir
func.func @single_device_switch(
```

**读法**：
- `@mesh` 是 `x=4`（**4 台设备**），而 `@single_dev_1`/`@single_dev_2` 是**单设备网格**。
- **用例名 `switch`** —— 程序在**两种网格之间切换**：
  一部分算子跑在 4 设备网格上、一部分跑在单设备网格上。
- **★ 这验证的是"网格切换时的 reshard"** ——
  从 `x=4` 切到单设备意味着**把所有分片收拢到一台设备**（需要通信）。
- 反之从单设备切回 `x=4` 意味着**广播**。

**★ 与 L1-01 的呼应**：
那里讲 `sdy.mesh` 的定义与 `device_ids`；本课看到**同一程序里多个网格并存**的实际用法。

---

## 四、`sdy_all_to_all_fully_scattered`：**通信优化**（全散射）

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
```

```mlir
// Resharding pattern:
```

```mlir
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
```

```mlir
// OptimizeCollectivesPass eliminates the collective_permute by decomposing
```

```mlir
// The permuted axes ("x" and "y") are fully scattered off dimension 0 to
```

**读法**（**注释完整描述了优化过程**）：
1. `// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on ...)`
   —— **初次降级**产生一个 `sdy.collective_permute`（交换 `x` 和 `y` 轴）。
2. `// OptimizeCollectivesPass eliminates the collective_permute by decomposing ...`
   —— **`OptimizeCollectivesPass` 把它消除**，方法是**分解**成别的通信。
3. `// The permuted axes ("x" and "y") are fully scattered off dimension 0 to ...`
   —— 被置换的轴 `x`/`y` **完全散射**（fully scattered）到第 0 维之外。

**★ 这是 L4-10 讲的"通信优化"的完整实例**：
> 一个昂贵的 `collective_permute` 被**分解**成更高效的 `all_to_all`。

**两个 RUN 行**（`"false"`/`"true"`）—— 验证 **REPL 与 HALO 两种模式结果一致**（L4-09）。

**★ `fully` vs `partially` 的区别**：
- **fully scattered**：`x` 和 `y` **两个轴都**从第 0 维散开
- **partially scattered**：**只有一个轴**散开（见下）

---

## 五、`sdy_all_to_all_partially_scattered`：**部分散射**

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"
```

```mlir
// RUN: %S/run_sdy_interpreter_test.sh %s %t "true"
```

```mlir
// Resharding pattern:
```

```mlir
// Initial lowering creates sdy.collective_permute (swapping "x" and "y" on
```

```mlir
// OptimizeCollectivesPass detects the collective_permute + all_to_all chain and
```

```mlir
// The permuted axes ("x" and "y") are partially scattered off dimension 0 (only
```

**读法**（**与 `fully` 版本逐句对比**）：
| 行 | `fully` 版本 | `partially` 版本 |
|---|---|---|
| 第 3 句 | `eliminates the collective_permute by **decomposing**` | `**detects** the collective_permute + all_to_all **chain** and` |
| 第 4 句 | `are **fully** scattered off dimension 0 **to**` | `are **partially** scattered off dimension 0 (**only**` |

**★ 两个版本的差异**：
- **`fully`**：`OptimizeCollectivesPass` **主动分解** `collective_permute`。
- **`partially`**：它**检测到**"`collective_permute` + `all_to_all` 链" ——
  即优化**依赖于已有的 `all_to_all`**，而不是凭空分解。
- 散射程度：`fully` 是两轴都散、`partially` 是**只有一部分**散开。

**★ 这验证了 L4-10 讲的"优化模式的多样性"**：
> 同一个 `collective_permute`，在不同上下文里可能被**完全消除**或**部分消除** ——
> 取决于周围有没有可复用的通信。

**两个文件都有两个 RUN 行** —— 都验证 REPL/HALO 等价。

---

## 六、四个文件的族谱

| 族 | 文件 | 验证什么 | 对应 pass |
|---|---|---|---|
| **单设备** | `single_device_add` | **空轴网格** `[]` | `resolve-single-device-sharding`（L4-14） |
| **单设备** | `single_device_switch` | 多设备 ↔ 单设备**切换** | 同上 |
| **通信优化** | `sdy_all_to_all_fully_scattered` | `collective_permute` **被完全分解** | `optimize-collectives`（L4-10） |
| **通信优化** | `sdy_all_to_all_partially_scattered` | **部分**分解（依赖已有 `all_to_all`） | 同上 |

**★ 本课的三条结论**：

1. **"空轴网格" `sdy.mesh @m = <[], device_ids=[0]>` 是单设备的表达方式** ——
   没有轴可分，但仍需一个合法 mesh。
2. **网格可以在一个程序里切换** —— 从多设备切到单设备意味着**收拢分片**（需要通信）。
3. **`collective_permute` 可以被优化消除** ——
   `fully` 是主动分解、`partially` 是复用已有的 `all_to_all`。

**一句话总结**：
> **L7-01 用四个端到端流水线测试展示了两个 pass 的实际效果** ——
> `resolve-single-device-sharding`（空轴网格与网格切换）与
> `optimize-collectives`（`collective_permute` 的完全/部分分解）。
>
> 这是"从导入到设备代码"这条时间轴上的**两个真实片段**。
