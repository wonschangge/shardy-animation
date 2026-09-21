# L5-02 · `lowering-sdy-collectives` — 集合通信降级（P0）

> 层：**L5 · 传播与降级** ｜ 优先级：P0 ｜ 前置课：`L4-08`、`L5-01`

## 学习目标

看完这一课，你应该能：

1. **写出某个 `sdy.all_gather` 对应的 `stablehlo.all_gather` 关键属性**（TODOLIST 验收点）；
2. 说出 `replica_groups` 的**两种表示**及 `enable-rgv3` 的作用；
3. 说出 **`all_slice` 为什么是唯一的例外**；
4. 说出归约类为什么需要 **reduction 区域**。

## 覆盖的测试文件（6 个 / 661 行 / 30 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/sdy_all_gather.mlir` | 176 | 7 |
| `convert_global_to_local/sdy_reduce_scatter.mlir` | 138 | 5 |
| `convert_global_to_local/sdy_collective_permute.mlir` | 104 | 6 |
| `convert_global_to_local/sdy_all_to_all.mlir` | 96 | 4 |
| `convert_global_to_local/sdy_all_slice.mlir` | 77 | 4 |
| `convert_global_to_local/sdy_all_reduce.mlir` | 70 | 4 |

## 场景（6 幕）

1. **★ 六条映射总表**
2. **★ `replica_groups` 的两种表示**
3. **★ `all_slice`：唯一的例外**
4. 归约类：为什么需要 reduction 区域
5. 其余两个：`all_to_all` 与 `collective_permute`
6. 30 个用例的族谱与五个共同点

## 核心结论

### ★ 六条映射

| SDY 算子 | 降级为 | 关键属性 |
|---|---|---|
| `sdy.all_gather` | `stablehlo.all_gather` | `all_gather_dim`、`replica_groups`、`use_global_device_ids` |
| `sdy.all_reduce` | `stablehlo.all_reduce` | + **reduction 区域** |
| **`sdy.all_slice`** | **无对应算子** → `partition_id` + `dynamic_slice` | 查找表 |
| `sdy.all_to_all` | `stablehlo.all_to_all` | `concat_dimension`、`split_count` |
| `sdy.reduce_scatter` | `stablehlo.reduce_scatter` | `scatter_dimension` + **reduction 区域** |
| `sdy.collective_permute` | `stablehlo.collective_permute` | `source_target_pairs` |

### ★ `replica_groups` 的两种表示

| | V1（`enable-rgv3=false`） | V3（**默认**） |
|---|---|---|
| 形式 | `dense<[[0,1,2,3],[4,5,6,7]]>` | `replica_group_mesh_axes<mesh=@mesh_2_4, axes=["y"]>` |
| 含义 | **设备号列表** | **mesh 轴名** |
| 特点 | 直观但依赖设备编号 | 更可读、**与设备号解耦** |

`all_gather.mlir` 有**四个 RUN 行** = `enable-rgv3` × `per-dim-all-gather` 的 2×2 组合。

**V1 的分组逻辑**：聚合 `y` 轴 → 同一个 `x` 值下的 4 台设备互相 gather。

### ★ `all_slice` 是唯一的例外

- **StableHLO 没有对应的切片通信算子** —— 因为 `all_slice` **不需要通信**！
- 语义是「每台设备留自己那片」→ 单机内就是一次**本地切片**。
- 实现：`partition_id` + 查找表 + `dynamic_slice`（与 L5-01 的常量处理同一套路）。
- **`partition_id`（模型并行）vs `replica_id`（数据并行）** —— L5-01 的常量用后者，本课用前者。

### 归约类需要 reduction 区域

`all_reduce` / `reduce_scatter` 都要**合并多个值**，但合并方式不唯一
（`add` / `max` / `min`…）→ 用一个**区域**指定。

### 五个共同点

① 局部类型都变小 ② `replica_groups` 两种表示 ③ 归约类带 reduction 区域
④ `channel_handle` 视模式而定（partition 有、replica 无）⑤ `all_slice` 是例外

### 与 L4-08 的完整链路

```
reshard  --L4-08-->  sdy.* 通信  --L5-02-->  stablehlo 通信
```

## 练习

见第 6 幕（本课练习内嵌在各幕的对照中）。

## 验收点

- [x] `check_ir_fidelity.py`：23 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：6 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能写出某个 `sdy.all_gather` 对应的 `stablehlo.all_gather` 关键属性
