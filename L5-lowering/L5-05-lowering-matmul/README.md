# L5-05 · `lowering-matmul` — 矩阵乘降级（P0）

> 层：**L5 · 传播与降级** ｜ 优先级：P0 ｜ 前置课：`L4-04`、`L4-05`、`L5-02`

## 学习目标

看完这一课，你应该能：

1. 说出**六种分片情形**分别对应哪种**经典并行策略**；
2. 说出**核心规律**：为什么只有收缩维上的分片需要 `all_reduce`；
3. 说出"延迟归约"在矩阵乘上的体现；
4. 说出 2D 并行中**哪一部分需要通信**。

## 覆盖的测试文件（2 个 / 119 行 / 6 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_dot.mlir` | 68 | 4 |
| `convert_global_to_local/stablehlo_dot_general.mlir` | 51 | 2 |

## 场景（6 幕）

1. **★ 六种情形 = 经典并行策略的 IR 体现**
2. 无通信的三种：各算各的
3. **★ 收缩维分片 = 数据并行**
4. **★ 情形 ④：延迟归约**
5. `dot_general`：批维 + 收缩维 = 2D 并行
6. 六种情形的小结与核心规律

## 核心结论

### ★ 六种情形总表

| 文件 | 用例 | 分片位置 | 通信 | 策略 |
|---|---|---|---|---|
| `dot` | `fully_replicated` | 无 | 无 | — |
| `dot` | `sharded_non_contracting_dims` | 非收缩维 | **无** | **模型并行** |
| `dot` | `sharded_contracting_dim` | 收缩维 | **`all_reduce`** | **数据并行** |
| `dot` | `..._unreduced_result` | 收缩维（结果未归约） | **无** | **延迟归约** |
| `dot_general` | `not_shard_contracting_dims` | **批维** | **无** | **批并行** |
| `dot_general` | `shard_contracting_dims` | 批维 + 收缩维 | **`all_reduce`** | **2D 并行** |

### ★ 核心规律

> **只有【收缩维】上的分片需要 `all_reduce`。**

**为什么**：
- **收缩维 = 归约的方向** → 沿它分片，每台设备只算了**部分和** → 必须合并
- **其他维 = 输出元素的方向** → 沿它分片，每台设备算的是**不同的输出元素** → 互不重叠

### 延迟归约（情形 ④）

与情形 ③ 的**唯一区别**：函数结果**也声明了 `unreduced`** → **不插 `all_reduce`**，
归约责任交给调用者。

**判据**：**收缩维是否被分片** + **结果是否接受未归约**。

### 完整链路

```
dot（收缩维分片）
  --L4-04-->  unreduced + sdy.all_reduce
  --L5-05-->  stablehlo.dot(unreduced) + stablehlo.all_reduce
```

## 验收点

- [x] `check_ir_fidelity.py`：67 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定分片位置，能说出对应哪种并行策略、需不需要通信
