<!-- sdy-coverage
transforms/export/test/executable_convert_global_to_local/stablehlo_all_gather.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_all_reduce.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_all_to_all.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_collective_permute.mlir
transforms/export/test/executable_convert_global_to_local/stablehlo_reduce_scatter.mlir
-->

# L6-02 · exec-stablehlo-collectives — 源 IR

**直接手写 `stablehlo` 集合通信时的数值验证** —— 与 L6-01 形成对照。

| 文件 | 行数 |
|---|---|
| `executable_convert_global_to_local/stablehlo_all_to_all.mlir` | 60 |
| `executable_convert_global_to_local/stablehlo_all_reduce.mlir` | 51 |
| `executable_convert_global_to_local/stablehlo_all_gather.mlir` | 48 |
| `executable_convert_global_to_local/stablehlo_collective_permute.mlir` | 45 |
| `executable_convert_global_to_local/stablehlo_reduce_scatter.mlir` | — |

网格（**单轴**）：

```mlir
sdy.mesh @mesh_4 = <["x"=4]>
```

---

## 一、★ 与 L6-01 的**关键差异**

**同样验证集合通信，但 part1 里写的东西完全不同**：

| | L6-01（`sdy_*`） | **L6-02（`stablehlo_*`）** |
|---|---|---|
| part1 里写的 | **`sdy.all_reduce`** | **`manual_computation` + `stablehlo.all_reduce`** |
| 抽象层次 | **高层**（SDY 算子） | **低层**（直接写硬件通信） |
| `replica_groups` | 由降级**生成** | **手写** |
| 对应课 | L5-02（降级） | **L4-11**（逐指令分区） |

**为什么两者都需要**：
- **L6-01** 验证"SDY 算子 → 降级 → 执行"这条路径
- **L6-02** 验证"手写底层通信 → 执行"这条路径
- 两者**语义应该等价** —— 因为 **L5-02 的降级产物就是 `stablehlo.*`**

---

## 二、★ 共同结构：`manual_computation` 包裹

5 个文件的 part1 **结构完全一致**：

```mlir
func.func @manual_all_reduce(
  %arg0: tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {}]>}) {
```

```mlir
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh_4, [{"x"}, {}]>]
    out_shardings=[<@mesh_4, [{}, {}]>]
    manual_axes={"x"} (%arg1: tensor<4x8xi32>) {
      %1 = "stablehlo.all_reduce"(%arg1) ({
        ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
          %2 = stablehlo.add %arg2, %arg3 : tensor<i32>
          stablehlo.return %2 : tensor<i32>
      }) {
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
        channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
        use_global_device_ids
      } : (tensor<4x8xi32>) -> tensor<4x8xi32>
      sdy.return %1 : tensor<4x8xi32>
  } : (tensor<16x8xi32>) -> (tensor<4x8xi32>)
  return %0 : tensor<4x8xi32>
}
```

**读法**（**这正是 L4-11 讲的"逐指令分区"形态**）：
- `sdy.manual_computation` 包裹 —— 区域内**自己管分片**（L1-07）。
- `manual_axes={"x"}` —— **`x` 轴冻结**，区域内不再分片。
- 区域内**直接放 `stablehlo.all_reduce`** —— 因为分片已经"手动处理"了。
- `in_shardings` / `out_shardings` 是**全局分片**；
  区域内 `%arg1` 是**局部形状**（`16x8` 沿 `x=4` 切 → `4x8`）✓

**`replica_groups = dense<[[0, 1, 2, 3]]>`**：
单轴 mesh（`@mesh_4`）→ **所有 4 台一组**，无需分组。

**`channel_handle = <handle = 1, type = 0>`**：
注意 **`type = 0`** —— 与 L5-02 降级产物（`type = 1`）**不同**。
这是**手写**的特征（降级生成的用 `type = 1`）。

---

## 三、`all_reduce`：4 个 shard 归约成 1111

```mlir
func.func @main() {
```

```mlir
  // Create distinct 4x8 local shards for the 4 virtual devices.
```

```mlir
  %s0 = stablehlo.constant dense<1> : tensor<4x8xi32>
```

```mlir
  %s1 = stablehlo.constant dense<10> : tensor<4x8xi32>
```

```mlir
  %s2 = stablehlo.constant dense<100> : tensor<4x8xi32>
```

```mlir
  %s3 = stablehlo.constant dense<1000> : tensor<4x8xi32>
```

```mlir
  // Expected global result for a sum reduction: 1 + 10 + 100 + 1000 = 1111.
```

```mlir
  %expected = stablehlo.constant dense<1111> : tensor<4x8xi32>
```

```mlir
  %res:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
```

```mlir
    programs = [[@manual_all_reduce, @manual_all_reduce, @manual_all_reduce, @manual_all_reduce]]
```

```mlir
  } : (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>) ->
      (tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>, tensor<4x8xi32>)
```

```mlir
  "check.expect_eq"(%res#0, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

```mlir
  "check.expect_eq"(%res#1, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

```mlir
  "check.expect_eq"(%res#2, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

```mlir
  "check.expect_eq"(%res#3, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

```mlir
  return
```

```mlir
}
```

**读法**（与 L6-01 的 `all_reduce` 用例**完全同构**）：
- 4 台设备各拿 `1/10/100/1000`。
- `replica_groups = [[0,1,2,3]]` → **全部 4 台一组**。
- 归约结果：`1 + 10 + 100 + 1000 = 1111`。
- **4 台设备都得到 `1111`** ✓

**与 L6-01 的对比**：
| | L6-01 | L6-02 |
|---|---|---|
| part1 写的 | `sdy.all_reduce {"x"}` | `stablehlo.all_reduce` + `replica_groups` |
| 期望值 | `dense<1111>` | `dense<1111>` |
| 结果 | 都得到 1111 | 都得到 1111 |

**★ 两者结果相同** —— 这验证了 **L5-02 的降级是语义保持的**。

---

## 四、`all_gather`：期望值**现算**

```mlir
func.func @main() {
```

```mlir
  // Create distinct 4x8 local shards for the 4 devices.
  %s0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  %s1 = stablehlo.constant dense<10.0> : tensor<4x8xf32>
  %s2 = stablehlo.constant dense<100.0> : tensor<4x8xf32>
  %s3 = stablehlo.constant dense<1000.0> : tensor<4x8xf32>

  // Create the expected global tensor for verification.
  %input = "stablehlo.concatenate"(%s0, %s1, %s2, %s3) {dimension = 0 : i64}
    : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<16x8xf32>
```

**读法**（**一个值得学的测试技巧**）：
- 期望值**不是硬编码的常量**，而是用 **`stablehlo.concatenate` 现算**出来的。
- 因为 `all_gather` 的语义就是"把各设备的 shard 沿某维**拼接**" ——
  用 `concatenate` 表达期望值**最直接、最不容易写错**。

**对比 `all_reduce`**：那里期望值是**硬编码的 `1111`** ——
因为"求和"的结果**无法用更简单的算子表达**（除非也写个 reduce）。

**★ 期望值的写法取决于算子语义**：
| 算子 | 期望值写法 |
|---|---|
| `all_reduce` | **硬编码常量**（`1111`）—— 求和结果最简单 |
| `all_gather` | **`concatenate` 现算** —— 语义就是拼接 |
| `all_to_all` | **`concatenate` 现算**（用置换后的输入） |

---

## 五、★ `collective_permute`：循环置换

```mlir
      %1 = "stablehlo.collective_permute"(%arg1) {
        source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>,
```

```mlir
  "check.expect_eq"(%res#0, %s3) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#1, %s0) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#2, %s1) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
  "check.expect_eq"(%res#3, %s2) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

**读法**（**本课最清晰的一个例子**）：
- `source_target_pairs = [[0,1], [1,2], [2,3], [3,0]]` —— 一个**循环置换**：
  设备 0 → 1、1 → 2、2 → 3、3 → **0**（回到起点）。
- **逐设备推结果**：
  | 设备 | 收到谁的 | 期望值 |
  |---|---|---|
  | 0 | 设备 **3** 发的 | `%s3` |
  | 1 | 设备 **0** 发的 | `%s0` |
  | 2 | 设备 **1** 发的 | `%s1` |
  | 3 | 设备 **2** 发的 | `%s2` |
- ✓ 与断言完全一致

**★ 这是"收发对照表"语义的最清晰展示**：
- `[0, 1]` 读作"**设备 0 发给设备 1**" —— 所以**设备 1 收到 `%s0`**。
- 断言写的是**接收方**的期望值 —— 注意**方向**。

**回顾 L5-02**：那里看到 `collective_permute` 的 `source_target_pairs`
且 `[1,4]`/`[4,1]` **成对出现**（交换语义）。
本课的 `[[0,1],[1,2],[2,3],[3,0]]` 是**循环**（不全是成对的）——
说明 `source_target_pairs` **不要求成对**，任意映射都行。

---

## 六、`reduce_scatter`：区域内输入是**完整**的

```mlir
func.func @manual_reduce_scatter(
```

```mlir
    manual_axes={"x"} (%arg1: tensor<16x8xi32>) {
```

```mlir
      %1 = "stablehlo.reduce_scatter"(%arg1) ({
```

```mlir
        replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
```

```mlir
  %expected = stablehlo.constant dense<1111> : tensor<4x8xi32>
```

```mlir
  "check.expect_eq"(%res#0, %expected) : (tensor<4x8xi32>, tensor<4x8xi32>) -> ()
```

**读法**（**与其他 4 个文件的一个关键区别**）：
- 区域内 `%arg1` 的类型是 **`tensor<16x8xi32>`** —— **完整大小**！
- 而 `all_reduce` 用例里区域内是 `tensor<4x8xi32>`（局部大小）。
- **为什么**：`reduce_scatter` 的 `in_shardings` 是 `[{}, {}]`（**全复制**）——
  区域内每台设备都有**完整数据**，然后 `reduce_scatter` 在区域内做"归约 + 切分"。

**结果**：4 台设备的输入都是完整的（各含 `1/10/100/1000` 的一部分？）——
从期望值 `1111` 看，归约后每台得到完整的和。

**★ 这验证了 L5-02 讲的 `reduce_scatter` 语义**：
"边归约边切分" —— 先在完整数据上归约，再切分给各设备。

---

## 七、5 个文件的族谱

| 文件 | 通信算子 | 期望值写法 | 结果分布 |
|---|---|---|---|
| `stablehlo_all_reduce` | `all_reduce` | 硬编码 `1111` | 所有设备相同 |
| `stablehlo_all_gather` | `all_gather` | **`concatenate` 现算** | 都得到完整拼接 |
| `stablehlo_all_to_all` | `all_to_all` | **`concatenate` 现算** | 置换后的拼接 |
| `stablehlo_collective_permute` | `collective_permute` | **各设备的输入** | **每台不同**（循环） |
| `stablehlo_reduce_scatter` | `reduce_scatter` | 硬编码 `1111` | 都得到完整的和 |

**★ 与 L6-01 的对照**：

| | L6-01（5 个 `sdy_*`） | L6-02（5 个 `stablehlo_*`） |
|---|---|---|
| 抽象 | **高层 SDY 算子** | **低层 stablehlo 算子** |
| 容器 | 无（直接写） | **`manual_computation`** |
| 分片处理 | 由 `sdy.*` 算子隐含 | **`manual_axes` 显式冻结** |
| `replica_groups` | 降级生成 | **手写** |
| 验证的路径 | L5-02 的**降级** | **手写底层通信** |

**★ 两者语义等价** —— 这正是 L5-02 降级的正确性依据。

**一句话总结**：
> **L6-01 验证"SDY 算子 → 降级 → 执行"，L6-02 验证"手写 stablehlo 通信 → 执行"。**
> 两者结果相同，证明 **L5-02 的降级是语义保持的**。
