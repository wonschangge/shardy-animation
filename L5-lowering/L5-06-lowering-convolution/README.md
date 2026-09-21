# L5-06 · `lowering-convolution` — 卷积降级

> 层：**L5 · 传播与降级** ｜ 优先级：P1 ｜ 前置课：`L2-02`、`L2-10`、`L5-05`

## 学习目标

看完这一课，你应该能：

1. 说出卷积的**六个用例**分别按哪个维度分片、是否需要通信；
2. 说出卷积的 **`sharding_rule`** 里哪些是归约因子；
3. 说出卷积比矩阵乘**多的三个复杂度**；
4. 说出子轴在卷积分片中的**实际用法**。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_convolution.mlir` | 191 | 6 |

## 场景（6 幕）

1. **★ 六个用例按分片的维度组织**
2. 无通信的两个：批维与特征维
3. **★ 归约因子分片：`unreduced` + `all_reduce`**
4. 情形 ⑥：延迟归约
5. 与 L5-05（矩阵乘）的对照
6. 六个用例的小结

## 核心结论

### 六个用例

| 用例 | 分片位置 | 通信 |
|---|---|---|
| `shard_batch` | 批维 | **无** |
| `shard_batch_group` | batch_group 维 | **无** |
| `shard_feature` | 特征维 | **无** |
| `shard_feature_group` | feature_group 维 | **无** |
| `shard__reduction_factors` | **归约因子** | **`all_reduce`** |
| `shard_reduction_factors_unreduced_result` | 归约因子（未归约） | **无**（延迟） |

### 卷积的 `sharding_rule`

```
([i, jk, mn, o], [k, n, o, p])->([i, j, m, p])
{i=2, j=112, k=2, m=112, n=2, o=3, p=64}
reduction={k, n, o} permutation={j, m}
```

- **`reduction={k, n, o}`** —— **窗口 + 输入通道**是归约因子
- **`permutation={j, m}`** —— 空间维因步长不成整数倍

### ★ 核心规律（与 L5-05 一致）

> **只有【归约因子】上的分片需要 `all_reduce`。**

**为什么**：`k`/`n`（窗口）与 `o`（输入通道）都是**被累加**的方向
→ 沿它们分片，每台设备只累加了**一部分** → 结果标 `unreduced`。

### 与矩阵乘的对照

| | 矩阵乘 | 卷积 |
|---|---|---|
| 输出维 | 非收缩维 | 批维 / 特征维 |
| 归约维 | 收缩维 | **窗口 + 输入通道** |
| 延迟归约 | 结果标 `unreduced` | 结果标 `unreduced` |

### 卷积多的三个复杂度

1. **归约因子有三个**（`k`/`n`/`o`）—— 组合更多
2. **`permutation` 因子** —— 带来 halo 问题（L4-09）
3. **`group_count`** 改变维度**大小** → 分片要用**子轴**精细分配

**子轴的实际用法**：`[{"x"}, {"y":(1)2}, {"y":(2)2}, {}]` ——
`y=4` 要**同时**切窗口维和输入通道维，用子轴 `(1)2`/`(2)2` 分成两份各切一个维。

## 验收点

- [x] `check_ir_fidelity.py`：25 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：源文件已声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定分片位置，能说出是否需要通信、为什么
