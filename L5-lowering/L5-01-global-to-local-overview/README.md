# L5-01 · `global-to-local-overview` — 全局转局部总览（L5 开篇）

> 层：**L5 · 传播与降级** ｜ 优先级：P0 ｜ 前置课：`L1-01`、`L4-15`

## 学习目标

看完这一课，你应该能：

1. **算出任意 sharding 对应的局部类型**（TODOLIST 验收点）；
2. 说出分片常量为什么必须用 `replica_id` + 运行时切片；
3. 解读查找表 `[0,0,0,0,2,2,2,2]` 的推导；
4. 说出四个选项的作用范围。

## 覆盖的测试文件（2 个 / 102 行 / 7 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/convert_global_to_local/generic_ops.mlir` | 83 | 6 |
| `transforms/export/test/convert_global_to_local/replica_id.mlir` | 19 | 1 |

## 场景（6 幕）

1. L5 开篇：从「描述分片」到「执行分片」
2. 类型转换：每个维度独立相除
3. **★ 常量：`replica_id` + 运行时切片**
4. 查找表 `[0,0,0,0,2,2,2,2]` 怎么来的
5. 四个选项
6. 练习

## 核心结论

### 转折点

| 层 | 张量是 | 视角 |
|---|---|---|
| L1～L4 | **全局的**（带分片标注） | 逻辑视图 |
| **L5+** | **局部的**（每台设备那份） | 物理视图 |

### 类型转换规则

> **维度大小 ÷ 该维度上轴的乘积**

**每个维度独立计算**：`x` 只影响它切的那一维，`y` 只影响它切的那一维。

| 张量 | 全局 | 分片 | 局部 |
|---|---|---|---|
| `%arg0` | `8x16` | `[{"x"}, {}]`（x=2） | **`4x16`** |
| `%arg1` | `16x32` | `[{}, {"y"}]`（y=4） | **`16x8`** |
| 结果 | `8x32` | `[{"x"}, {"y"}]` | **`4x8`** |

**三个要点**：
- **无分片的类型不变**（如 `tensor<i1>` 谓词）。
- **未注册方言也转换** —— 转换是「**按值的类型**」驱动的（所以要 `-allow-unregistered-dialect`）。
- **递归处理**：区域算子内部、函数调用两端都改。

### ★ 分片常量：`replica_id` + 运行时切片（六步）

1. **保留全局常量**（所有设备都有完整数据）
2. `stablehlo.replica_id` 取当前设备号
3. `stablehlo.convert` → `i64`（类型对齐）
4. **查找表** `dense<[0, 0, 0, 0, 2, 2, 2, 2]>`（8 台设备各自的**起始行号**）
5. `dynamic_slice` 查表取出「我的起始行」
6. `dynamic_slice` 从全局常量切出**本地那 2 行**

**查找表的推导**：
- `x` 有 2 个值切 `tensor<4>` → 各 2 行 → `x=0` 起始 0、`x=1` 起始 2。
- 设备号按轴序**最 major 优先**展开（`x` 在前 → 变化最慢）→ 设备 0~3 是 `x=0`、设备 4~7 是 `x=1`。
- → `[0,0,0,0,2,2,2,2]`。

**代价与收益**：每台设备持有完整常量 + 两次 `dynamic_slice`，换来**语义正确**。

### 四个选项（说明逐字来自 `sdy_opt --help`）

| 选项 | 说明 |
|---|---|
| `replica-count=<long>` | Number of replicas (**data parallelism**) |
| `partition-count=<long>` | Number of partitions (**model parallelism**) |
| `enable-rgv3` | Use StableHLO **ReplicaGroupV3** (mesh-axes based) for collectives |
| `per-dim-all-gather` | Keep **per-dimension** all-gather without combining them into a single all-gather |

后两个不在本课的 2 个文件里 —— 属后续课程（L5-02 等）。
**不要凭名字猜语义**（这正是 `check_flags.py` 存在的意义）。

## 练习

见第 6 幕。三道题分别考类型转换、常量处理、查找表。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag（四个选项均实测存在）
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能算出任意 sharding 对应的局部类型
