<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/run_sdy_interpreter_test.sh
transforms/export/test/executable_partitioner_pipeline/run_sdy_interpreter_test.sh
-->

# L6-00 · executable-test-mechanism — 机制说明

**本课不讲 IR，讲机制** —— Shardy 的"可执行测试"是怎么工作的。

| 文件 | 作用 |
|---|---|
| `transforms/export/test/executable_convert_global_to_local/run_sdy_interpreter_test.sh` | 导出流水线的执行测试 |
| `transforms/export/test/executable_partitioner_pipeline/run_sdy_interpreter_test.sh` | **完整分区器流水线**的执行测试 |

**为什么需要这一课**：
L4/L5 的许多测试文件用的是 `// RUN: %S/run_sdy_interpreter_test.sh %s %t`
（如 L3-11、L4-01 见过的 `executable_*`）。
它们**不是比对 IR 文本**，而是**真的执行**。
不理解这个机制，就看不懂那一大批测试。

---

## 一、★ 完整流程

### 脚本的参数解析

```bash
set -e

SRC=""
TMP=""
# When replica_count=partition_count=1, the test uses partition id.
REPLICA_COUNT=1
PARTITION_COUNT=1

pos_idx=0

for arg in "$@"; do
  case "$arg" in
    --src=*)                  SRC="${arg#*=}" ;;
    --temp_dir=*)             TMP="${arg#*=}" ;;
    --replica_count=*)        REPLICA_COUNT="${arg#*=}" ;;
    --partition_count=*)      PARTITION_COUNT="${arg#*=}" ;;
    --*)                      echo "Warning: Unknown flag '$arg'" >&2 ;;
    *)
      pos_idx=$((pos_idx + 1))
      case "$pos_idx" in
        1) SRC="$arg" ;;
        2) TMP="$arg" ;;
        3) REPLICA_COUNT="$arg" ;;
        4) PARTITION_COUNT="$arg" ;;
      esac
      ;;
  esac
done
```

**读法**：
- 支持**两种**传参方式：`--src=...` 命名参数，或**位置参数**（第 1 个是 src、第 2 个是 temp_dir）。
- 所以 `// RUN: %S/run_sdy_interpreter_test.sh %s %t` 用的是**位置参数**：
  `%s` = 源文件、`%t` = 临时目录（lit 的内置变量）。
- **默认** `REPLICA_COUNT=1 PARTITION_COUNT=1` —— 注释说
  "When replica_count=partition_count=1, the test uses partition id"
  （单设备情形下用 `partition_id` 而非 `replica_id` —— 与 **L5-01** 的常量处理呼应）。

### 工具的可覆盖性

```bash
SPLIT_FILE=${SPLIT_FILE:-split-file}
SDY_OPT=${SDY_OPT:-sdy_opt}
STABLEHLO_TRANSLATE=${STABLEHLO_TRANSLATE:-stablehlo-translate}
```

**读法**：三个工具都用 `${VAR:-default}` 形式 ——
**可以用环境变量覆盖**（便于测试基础设施替换实现）。

### 四步流程

```bash
"$SPLIT_FILE" "$SRC" "$TMP"
"$SDY_OPT" "$TMP/part1.mlir" \
  --sdy-convert-global-to-local="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
  --sdy-inline-meshes \
  --sdy-drop-sharding-and-mesh \
  --allow-unregistered-dialect > "$TMP/part1_processed.mlir"
sed '1d; /^}/,$d' "$TMP/part1_processed.mlir" > "$TMP/combined.mlir"
```

**逐步读**：

| 步 | 命令 | 作用 |
|---|---|---|
| ① | `split-file "$SRC" "$TMP"` | 按 `//--- partN.mlir` 标记**拆分**源文件 |
| ② | `sdy_opt part1.mlir --sdy-convert-global-to-local=...` | 跑**导出流水线** |
| ③ | `--sdy-inline-meshes --sdy-drop-sharding-and-mesh` | **清理**（L4-15 讲过） |
| ④ | `sed '1d; /^}/,$d'` | **提取函数体** |

**第 ④ 步的 `sed` 值得细看**：
- `1d` —— 删掉第 1 行
- `/^}/,$d` —— 从第一个 `}` 开始**删到文件尾**

**为什么这样能提取函数体**：`sdy_opt` 的输出是
```
module {
  func.func @parallel_x(...) {
    ...函数体...
  }
}
```
删掉第 1 行（`module {`）、删掉从 `}` 到末尾（函数结束的 `}` + module 的 `}`）
→ **只剩函数体** ✓

### ★ 串行参考实现

```bash
# If part1.mlir contains @parallel_x but not @sequential_x, then remove sharding
# from @parallel_x and rename it to @sequential_x.
if (grep -q "@parallel_" "$TMP/part1.mlir") && (! grep -q "@sequential_" "$TMP/part1.mlir"); then
  "$SDY_OPT" "$TMP/part1.mlir" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect | \
  sed 's/parallel_/sequential_/g' > "$TMP/part1_sequential.mlir"
  sed '1d; /^}/,$d' "$TMP/part1_sequential.mlir" >> "$TMP/combined.mlir"
fi
```

**读法**（**本课最关键的一步**）：
- 如果 `part1.mlir` 里有 `@parallel_` 但**没有** `@sequential_`：
  1. 把 `part1.mlir` 跑一遍 `--sdy-drop-sharding-and-mesh`
     —— **去掉所有分片信息** → 得到**串行版本**
  2. `sed 's/parallel_/sequential_/g'` —— **改名**（`@parallel_x` → `@sequential_x`）
  3. 提取函数体，**追加**到 `combined.mlir`

**这就是"参考实现"的来源**：
- `@parallel_x` 是**分片版**（经过导出流水线 → 局部代码）
- `@sequential_x` 是**同一个函数去掉分片** → **串行计算**
- 两者**语义应该等价** —— 只是并行方式不同

### 执行与验证

```bash
cat "$TMP/part2.mlir" >> "$TMP/combined.mlir"
"$STABLEHLO_TRANSLATE" --interpret "$TMP/combined.mlir"
```

**读法**：
- `part2.mlir` 追加到 `combined.mlir` —— 它里面有一个 `main` 函数，
  **同时调用** `@parallel_x` 与 `@sequential_x`，并**比较结果**。
- `stablehlo-translate --interpret` —— **真正执行**。

**★ 验证思路的完整链条**：
```
part1.mlir（分片版）
  --导出流水线--> 局部代码（@parallel_x）
  --drop-sharding--> 串行代码（@sequential_x）
                    ↓
              part2.mlir 的 main 同时调用两者、比较结果
                    ↓
              --interpret 真正执行 → 数值一致 = 语义保持 ✓
```

**为什么必须执行**（回顾 L3-11 / L4-01 的讨论）：
这些 pass **改写的是算子的语义属性**（分片、通信、形状）。
改错了 IR **依然能通过校验**，但**跑出来的数值会错**。
只有真正执行一遍才能确认语义保持。

---

## 二、★ `partitioner_pipeline` 版本：完整流水线

第二个脚本与第一个的**关键区别**：它跑的是**完整的分区器流水线**。

```bash
ENABLE_HALO_EXCHANGE=true
# When replica_count=partition_count=1, the test use partition id.
```

**多了一个参数** `ENABLE_HALO_EXCHANGE`（默认 `true`）——
对应 **L4-09** 的 `enable-halo-exchange` 选项。

### 它跑的 pass 序列（本课最有价值的一处）

```bash
# Run the partitioner pipeline passes.
  --sdy-insert-explicit-reshards="enable-full-version=true mark-partial-result-with-unreduced-axes=true" \
  --sdy-resolve-permutation-factors="enable-halo-exchange=$ENABLE_HALO_EXCHANGE replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
  --sdy-reshard-to-collectives \
  --sdy-optimize-collectives \
  --sdy-pad-for-divisibility \
  --sdy-resolve-single-device-sharding="replica-count=$REPLICA_COUNT partition-count=$PARTITION_COUNT" \
```

**逐行对应到课**：

| pass | 对应课 |
|---|---|
| `--sdy-insert-explicit-reshards` | **L4-02～07**（reshard 插入总纲与算子族） |
| `--sdy-resolve-permutation-factors` | **L4-09**（置换因子消解） |
| `--sdy-reshard-to-collectives` | **L4-08**（reshard 转集合通信） |
| `--sdy-optimize-collectives` | **L4-10**（通信优化） |
| `--sdy-pad-for-divisibility` | **L5-09**（为整除性补齐） |
| `--sdy-resolve-single-device-sharding` | **L4-14**（单设备分片） |

**★ 这就是 L4 + L5 的完整流水线** ——
把 17 + 9 课讲的 pass **串成了一条真实可跑的命令**。

**注意选项的传递**：
- `enable-full-version=true`（L4-02 见过）
- `mark-partial-result-with-unreduced-axes=true`（L4-05 见过）
- `enable-halo-exchange=...`（L4-09 见过）
- `replica-count` / `partition-count`（L5-01 见过）

**所有选项都在前面的课里出现过** —— 本课是它们的**集成视图**。

---

## 三、如何自己写一个可执行测试

**验收点**：能独立写出一个最小可执行测试并通过。

### 模板

```text
// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

func.func @parallel_add(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>},
                        %arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
       : tensor<4xf32>
  return %0 : tensor<4xf32>
}

//--- part2.mlir

func.func @main() {
  // 构造输入、调用 @parallel_add 与 @sequential_add、比较结果
  ...
}
```

### 三个必要元素

| 元素 | 要求 |
|---|---|
| **`// RUN:` 行** | 用 `%S/run_sdy_interpreter_test.sh %s %t` |
| **`part1.mlir`** | 含 `@parallel_*` 函数（**带分片**），至少一个 |
| **`part2.mlir`** | 含 `@main`，**同时调用** parallel 与 sequential 版本并比较 |

### 命名约定（关键）

- 分片版函数名必须含 **`@parallel_`** 前缀
- 脚本会**自动生成** `@sequential_` 版本（去分片 + 改名）
- 所以 `part2.mlir` 里要调用 **两个** 名字：`@parallel_x` 与 `@sequential_x`

**注意**：如果 `part1.mlir` 里**已经**有 `@sequential_`，
脚本就**不会**自动生成（`if` 条件要求"有 parallel 且**没有** sequential"）——
这是给"手写参考实现"留的口子。

### 一个最小例子

```text
// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

func.func @parallel_negate(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
       : tensor<8xf32>
  return %0 : tensor<8xf32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf32>
  %p = call @parallel_negate(%input) : (tensor<8xf32>) -> tensor<8xf32>
  %s = call @sequential_negate(%input) : (tensor<8xf32>) -> tensor<8xf32>
  // 比较 %p 与 %s ...
  return
}
```

**流程**：
1. `split-file` 拆出 `part1.mlir` / `part2.mlir`
2. 导出流水线处理 `part1` → `@parallel_negate` 变成**局部代码**
3. `drop-sharding` + 改名 → `@sequential_negate`
4. `main` 同时调用两者
5. `--interpret` 执行 → 结果一致 ✓

---

## 四、本课在 L6 中的位置

**L6 是执行与解释器层**（8 课），本课是**开篇**：

| 课 | 主题 |
|---|---|
| **L6-00（本课）** | **可执行测试的机制** |
| L6-01 | `sdy.*` 集合通信的执行 |
| L6-02 | `stablehlo` 集合通信的执行 |
| L6-03 | 卷积的执行 |
| L6-04 | 矩阵乘 / fft / iota 的执行 |
| L6-05 | gather 的执行（★ 最复杂） |
| L6-06 | pad 的执行（★ 本层最大族） |
| L6-07 | reshape 的执行 |
| L6-08 | reverse / slice 的执行 |
| L6-09 | scatter 与杂项 |

**后续所有课都用本课讲的机制** —— 它们都是 `executable_*` 目录下的测试，
用 `run_sdy_interpreter_test.sh` 驱动。

**一句话总结**：
> **Shardy 的可执行测试 = "分片版 vs 串行版"的结果对比。**
> 脚本自动从分片版**生成**串行版（去分片 + 改名），
> 再由 `main` 同时调用两者、用 `--interpret` **真正执行**来验证语义保持。
