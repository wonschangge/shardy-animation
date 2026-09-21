# L6-00 · `executable-test-mechanism` — 可执行测试机制（P0 · L6 开篇）

> 层：**L6 · 执行与解释器** ｜ 优先级：P0 ｜ 前置课：`L4-15`、`L5-01`、`L5-09`

## 学习目标

看完这一课，你应该能：

1. **独立写出一个最小可执行测试并通过**（TODOLIST 验收点）；
2. 说出可执行测试的**验证思路**（分片版 vs 串行版）；
3. 说出**串行参考版是怎么来的**；
4. 说出第二个脚本跑的**完整流水线**对应哪些课。

## 覆盖的文件（2 个 shell 脚本，**非 IR**）

| 文件 | 作用 |
|---|---|
| `executable_convert_global_to_local/run_sdy_interpreter_test.sh` | 导出流水线的执行测试 |
| `executable_partitioner_pipeline/run_sdy_interpreter_test.sh` | **完整分区器流水线**的执行测试 |

> **注意**：它们是 `.sh` 而非 `.mlir` —— **不计入 241 个测试 IR**，
> 所以本课对覆盖率的贡献是 **0**。这是**诚实**的：TODOLIST 也说这是「机制说明，非 IR」。

## 场景（6 幕）

1. L6 开篇：为什么 Shardy 的测试能「跑出数值」
2. **★ 脚本的四步流程**
3. **★ 最关键的机制：自动生成串行参考版**
4. **★ 第二个脚本：完整分区器流水线**
5. 如何自己写一个可执行测试
6. 小结与 L6 全层预览

## 核心结论

### ★ 验证思路

```
part1.mlir（分片版）
  --导出流水线--> 局部代码（@parallel_x）
  --drop-sharding--> 串行代码（@sequential_x）
                    ↓
              part2.mlir 的 main 同时调用两者、比较结果
                    ↓
              --interpret 真正执行 → 数值一致 = 语义保持 ✓
```

**为什么必须执行**：这些 pass 改写的是**语义属性**（分片、通信、形状）。
改错了 IR **依然能通过校验**，但**跑出来的数值会错**。

### ★ 串行参考版怎么来的

```bash
if (grep -q "@parallel_" "$TMP/part1.mlir") && (! grep -q "@sequential_" "$TMP/part1.mlir"); then
  "$SDY_OPT" "$TMP/part1.mlir" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
  sed 's/parallel_/sequential_/g' > "$TMP/part1_sequential.mlir"
```

**三步**：① `--sdy-drop-sharding-and-mesh` 去掉所有分片 → 串行语义；
② `sed` 改名 `@parallel_x` → `@sequential_x`；③ 提取函数体追加。

**为什么 `drop-sharding` 就是串行版**：删掉分片后 IR 变成**纯 StableHLO**
—— 即「单设备上直接算」，这正是**串行语义**的定义。

**一个细节**：`if` 条件要求「有 parallel 且**没有** sequential」——
给**手写参考实现**留的口子。

### ★ 完整分区器流水线

第二个脚本跑的 pass 序列，**每一行都对应一课**：

| pass | 对应课 |
|---|---|
| `--sdy-insert-explicit-reshards` | **L4-02～07** |
| `--sdy-resolve-permutation-factors` | **L4-09** |
| `--sdy-reshard-to-collectives` | **L4-08** |
| `--sdy-optimize-collectives` | **L4-10** |
| `--sdy-pad-for-divisibility` | **L5-09** |
| `--sdy-resolve-single-device-sharding` | **L4-14** |

→ **这是 L4 + L5 共 26 课的集成视图**，也是一条真实可跑的命令。

### 四步流程

| 步 | 命令 | 作用 |
|---|---|---|
| ① | `split-file` | 按 `//--- partN.mlir` 拆分 |
| ② | `sdy_opt` | 跑导出流水线 |
| ③ | `sed '1d; /^}/,$d'` | **提取函数体** |
| ④ | `stablehlo-translate --interpret` | **真正执行** |

### 自己写一个：三个必要元素

1. `// RUN: %S/run_sdy_interpreter_test.sh %s %t`
2. `part1.mlir`：含 **`@parallel_*`** 函数（带分片）
3. `part2.mlir`：含 `@main`，**同时调用**两个版本并比较

**建议起点**：用 `negate` 这类**逐元素算子** —— 分片后无需通信，最容易写对。

## 验收点

- [x] `check_ir_fidelity.py`：0 个 IR 块（本课为机制说明，无真实 IR）
- [x] `check_coverage.py`：2 个 `.sh` 文件已声明（不计入 241）
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能写出「三元素 + 命名约定」齐备的最小可执行测试
