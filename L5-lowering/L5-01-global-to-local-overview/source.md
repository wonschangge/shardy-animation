<!-- sdy-coverage
transforms/export/test/convert_global_to_local/generic_ops.mlir
transforms/export/test/convert_global_to_local/replica_id.mlir
-->

# L5-01 · global-to-local-overview — 源 IR

**L5 层的开篇，也是整个动画的一个转折点**：
前面 L1～L4 讲的都是"**分片怎么流动**"；从本课开始讲"**分片怎么变成每台设备实际持有的数据**"。

| 文件 | 行数 | 用例数 | RUN 行 |
|---|---|---|---|
| `transforms/export/test/convert_global_to_local/generic_ops.mlir` | 83 | 6 | `-sdy-convert-global-to-local -allow-unregistered-dialect` |
| `transforms/export/test/convert_global_to_local/replica_id.mlir` | 19 | 1 | `-sdy-convert-global-to-local="replica-count=8 partition-count=1"` |

网格：

```mlir
sdy.mesh @mesh_2 = <["x"=2]>
```

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

---

## 一、★ 核心思路：全局张量 → 局部张量

### 类型转换

```mlir
func.func @func_returning_sharded_arg(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
```

```mlir
  return %arg0 : tensor<16xf32>
}
```

```mlir
// CHECK-LABEL: func.func @func_returning_sharded_arg
// CHECK-SAME:    (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
```

```mlir
  // CHECK-NEXT:  return %arg0 : tensor<8xf32>
```

**读法**（本课最基本的一条规则）：
- 全局 `tensor<16xf32>` 在 `@mesh_2`（`x=2`）上切 `{"x"}` → **局部 `tensor<8xf32>`**
- **16 ÷ 2 = 8** ✓
- **分片属性保留不变**（`[{"x"}]`）—— 它记录"这个局部张量是怎么来的"。

### 所有算子都改写成局部形状

```mlir
func.func @func_with_dot_then_add(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>},
  %arg2: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
```


```mlir
  // CHECK-NEXT:  %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[DOT]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<4x8xf32>
  %1 = stablehlo.add %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<8x32xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<4x8xf32>
  return %1 : tensor<8x32xf32>
}
```

**逐项读**（`@mesh_2_4` 是 `x=2, y=4`）：

| 张量 | 全局 | 分片 | 局部 | 算式 |
|---|---|---|---|---|
| `%arg0` | `8x16` | `[{"x"}, {}]` | **`4x16`** | 8 ÷ 2 = 4 |
| `%arg1` | `16x32` | `[{}, {"y"}]` | **`16x8`** | 32 ÷ 4 = 8 |
| `dot` 结果 | `8x32` | `[{"x"}, {"y"}]` | **`4x8`** | 8÷2=4, 32÷4=8 |
| `%arg2` | `8x32` | `[{"x"}, {"y"}]` | **`4x8`** | 同上 |

**关键观察**：**每个维度独立地按它对应的轴大小相除** ——
`x` 只影响第 0 维（÷2），`y` 只影响第 1 维（÷4）。

### 未注册方言也要转换

```mlir
func.func @unknown_dialect_op(
  %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
  -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
```

```

```mlir
  // CHECK-NEXT:  "interpreter.print"(%[[ARG0]]) : (tensor<8xf32>) -> ()
  "interpreter.print"(%arg0) : (tensor<16xf32>) -> ()
  // CHECK-NEXT:  return %[[ARG0]] : tensor<8xf32>
  return %arg0 : tensor<16xf32>
}
```

**读法**：`interpreter.print` 是**未注册方言**的算子 —— 但它的**操作数类型**同样被改成 `tensor<8xf32>`。

**这说明类型转换是"按值的类型"驱动的，与算子是否注册无关** ——
所以 RUN 行要加 `-allow-unregistered-dialect`。

### 区域算子内部同样处理

```mlir
func.func @stablehlo_while_sharded(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
```

```mlir
  %0 = stablehlo.while(%iterArg = %arg0) : tensor<16xf32>
    attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>}
    cond {
      %pred = stablehlo.constant dense<false> : tensor<i1>
      stablehlo.return %pred : tensor<i1>
    } do {
      %add = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<16xf32>
      stablehlo.return %add : tensor<16xf32>
    }
```

```mlir
  return %0 : tensor<16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[WHILE:.*]] = stablehlo.while(%iterArg = %arg0) : tensor<8xf32>
  // CHECK-NEXT:  cond {
  // CHECK-NEXT:    %[[PRED:.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-NEXT:    stablehlo.return %[[PRED]] : tensor<i1>
  // CHECK-NEXT:  } do {
  // CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : tensor<8xf32>
  // CHECK-NEXT:    stablehlo.return %[[ADD]] : tensor<8xf32>
  // CHECK-NEXT:  }
```

**读法**：`while` 的**循环变量类型**从 `tensor<16xf32>` 变成 `tensor<8xf32>`，
**区域内部**的 `add` 也一样。

**注意**：`%[[PRED]]` 的类型 `tensor<i1>` **不变** ——
因为它是**无分片**的（i1 是谓词，不在网格上切）。

**规则**：**只有带分片的张量类型才转换**；无分片的（如 `tensor<i1>`、标量）保持原样。

### 函数调用

```mlir
func.func @func_call_sharded(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
```

```mlir
  %0 = call @callee_sharded(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : (tensor<16xf32>) -> tensor<16xf32>
```

```mlir
  return %0 : tensor<16xf32>
}
```

```mlir
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_sharded(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
```

**读法**：`call` 的**实参类型与结果类型**都变成局部 —— 被调函数的签名也一样（本文件里 `@callee_sharded` 的定义同样被转换）。

---

## 二、★ 常量怎么处理：`replica_id` + 运行时切片

**这是本课最精妙的部分。**

### 问题

全局常量 `dense<[[1,2],[3,4],[5,6],[7,8]]>` 是 `tensor<4x2xf32>`，
在 `@mesh_2_4`（`x=2, y=4`，共 8 台设备）上切 `[{"x"}, {}]`。

**局部类型**是 `tensor<2x2xf32>` —— 但**每台设备需要不同的 2 行**！
- 设备 0、1 需要第 0~1 行
- 设备 2、3 需要第 2~3 行
- ……

**怎么表达"每台设备拿不同的常量切片"？**

### 解法

```mlir
// RUN: sdy_opt %s -sdy-convert-global-to-local="replica-count=8 partition-count=1" | FileCheck %s
```

```mlir
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
```

```mlir
func.func @sharded_dense_replica_id() -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
```

```mlir
  %0 = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>
```

```mlir
  // CHECK-NEXT: %[[GLOBAL_CST:.*]] = stablehlo.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[RID:.*]] = stablehlo.replica_id : tensor<ui32>
  // CHECK-NEXT: %[[RID_I64:.*]] = stablehlo.convert %[[RID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 2, 2, 2, 2]> : tensor<8xi64>
  // CHECK-NEXT: %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[RID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[START_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[START_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[LOCAL_SLICE:.*]] = stablehlo.dynamic_slice %[[GLOBAL_CST]], %[[START_0]], %[[START_1]], sizes = [2, 2] : (tensor<4x2xf32>, tensor<i64>, tensor<i64>) -> tensor<2x2xf32>
```

```mlir
  // CHECK-NEXT: return %[[LOCAL_SLICE]] : tensor<2x2xf32>
```

### 逐项读（六步）

| 步 | 操作 | 作用 |
|---|---|---|
| ① | **保留全局常量** `tensor<4x2xf32>` | 所有设备都有完整数据 |
| ② | `stablehlo.replica_id` | 取**当前设备号**（0~7） |
| ③ | `stablehlo.convert` → `i64` | 类型对齐（索引用 i64） |
| ④ | **查找表** `dense<[0, 0, 0, 0, 2, 2, 2, 2]>` | 8 台设备 → 各自的**起始行号** |
| ⑤ | `dynamic_slice` 查表 | 取出"我的起始行" |
| ⑥ | `dynamic_slice` 切常量 | 从全局常量切出**本地那 2 行** |

**查找表的含义**（`[0,0,0,0,2,2,2,2]`）：
- 设备 0~3 → 起始行 **0**
- 设备 4~7 → 起始行 **2**

**为什么是 4 个一组**：`@mesh_2_4` 的轴序是 `["x"=2, "y"=4]`，
分片 `[{"x"}, {}]` 只切 `x`。设备号按 `x` **最 major** 排列：
`x` 取 0 时对应设备 0~3，`x` 取 1 时对应设备 4~7。
而 `x` 只有 2 个值，切 `tensor<4>` 的第 0 维 → 每个 `x` 值拿 2 行。
所以设备 0~3（`x=0`）起始行 0，设备 4~7（`x=1`）起始行 2 ✓

### 为什么这样做

**问题**：`sdy.constant` 在每台设备上**内容不同**（各拿一片），
而 `stablehlo.constant` 是**所有设备相同**的。

**解法**：保留全局常量（所有设备都有），再用 **`replica_id` 在运行时切出自己那份**。

**代价**：每台设备都要**持有完整常量** + 多做两次 `dynamic_slice`。
**收益**：语义正确 —— 每台设备确实拿到自己该拿的那片。

> **这就是 `replica_id` 文件的核心价值**：
> 它演示了"**分片常量**"如何在没有专门算子的情况下表达出来。

---

## 三、四个选项

| 选项 | 说明（来自 `sdy_opt --help`） |
|---|---|
| `replica-count=<long>` | Number of replicas (**data parallelism**) |
| `partition-count=<long>` | Number of partitions (**model parallelism**) |
| `enable-rgv3` | Use StableHLO **ReplicaGroupV3** (mesh-axes based) for collectives |
| `per-dim-all-gather` | Keep **per-dimension** all-gather without combining them into a single all-gather |

**读法**：
- **`replica-count` / `partition-count`** 是两个**基本参数** ——
  它们告诉转换器"总共多少台设备、怎么划分"。
  `replica-count=8 partition-count=1` 表示"8 个副本、1 个分区"。
  这与 **L1-01 的 `@maximal_mesh`** 概念相关：局部类型的大小取决于总设备数。
- **`enable-rgv3`** 影响**集合通信的表示**（用 mesh 轴而非设备号列表）——
  在 L5-02 会看到。
- **`per-dim-all-gather`** 影响**通信的合并策略** ——
  默认会把多维 all-gather 合并成一个；打开则保持逐维。

**这两个"高级"选项不在本课的 2 个文件里** —— 它们属于后续课程（L5-02 等）。
本课只需知道它们**存在**且**作用范围**如上。

---

## 四、本课在 L5 中的位置

**L5 是传播与降级层**（9 课），本课是**开篇总览**：

| 课 | 主题 |
|---|---|
| **L5-01（本课）** | **全局 → 局部的总体思路** |
| L5-02 | 集合通信算子的降级 |
| L5-03 | 结构性算子的降级 |
| L5-04 | 逐元素与形状类的降级 |
| L5-05 | 矩阵乘的降级 |
| L5-06 | 卷积的降级 |
| L5-07 | 归约的降级 |
| L5-08 | gather/scatter 的降级 |
| L5-09 | 为整除性加 padding |

**一句话总结**：
> **`convert_global_to_local` 把"全局张量 + 分片"变成"每台设备实际持有的局部张量"。**
> 类型按轴大小相除、算子改写成局部形状、常量用 `replica_id` + 运行时切片。
>
> 这是从"**描述分片**"到"**执行分片**"的关键一步。
