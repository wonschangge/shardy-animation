# L4-06 · `reshard-memory-ops` — 访存与通信类的 reshard

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L3-11`、`L4-04`

## 学习目标

看完这一课，你应该能：

1. 说出 `gather` 的处理**与 `dot` 同构**及原因；
2. 说出 L3-11（可表达）与 L4-06（可落地）的**呼应关系**；
3. 说出用户 `custom` 规则与内置注册规则的**同等地位**；
4. 说出**集合通信算子自身也有分片规则**，以及 `need_replication` 的来源。

## 覆盖的测试文件（3 个 / 713 行 / 39 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `insert_explicit_reshards/gather_scatter.mlir` | 288 | 9 |
| `insert_explicit_reshards/custom_call.mlir` | 318 | 24 |
| `insert_explicit_reshards/collective_ops.mlir` | 107 | 6 |

## 场景（6 幕）

1. **`gather`：与 `dot` 同构的四步**
2. 与 L3-11 的呼应：先「可表达」，再「可落地」
3. `custom_call`：内置注册 vs 用户规则，同等对待
4. **★ 集合通信算子自身也有分片规则**
5. 3 个文件 / 39 个用例的族谱
6. 练习

## 核心结论

### `gather` 与 `dot` 同构

四步链条（测试称其为 "the most expressive example"）：
1. 两个操作数各 **reshard**（把冲突的轴去掉/对齐）
2. `gather` 执行，结果标 **`unreduced`**（部分结果）
3. **`all_reduce`** 合并
4. **再 reshard** 到目标分片

**为什么同构**：两者的规则里都有 **`reduction` 因子**
（`gather` 是 `reduction={i, p}`）—— 沿归约因子切会产生部分结果。

该用例大量使用**子轴**（`{"x":(1)2}`、`{"y":(2)2}`…），因为 `@mesh_xyzt` 有 512 台设备。

### 与 L3-11 的呼应

- **L3-11（导入期）**：把 `iota` 从 `start_index_map` 移到 `operand_batching_dims`
  → 分片**可表达**（不再落在 `blocked_propagation` 因子上）。
- **L4-06（导出期）**：显式化之后，分片能被正常处理 → 插 reshard + `all_reduce`
  → 分片**可落地**。

**顺序不能反**：没有 L3-11 的显式化，L4-06 的这些 reshard 根本无从谈起。

### `custom_call`

- **22 个内置注册算子**（与 L2-10 一致）：按注册表的规则判定兼容性。
- **未注册 + 用户规则**：**与内置规则同等对待**。
  证据：`unregistered_custom_call_with_existing_rule` 里 `@foo` 未注册，
  但用户的转置规则让输入输出**完全兼容** → `CHECK-NOT: sdy.reshard`。

### ★ 集合通信算子自身也有分片规则

以 `all_gather` 为例（规则由测试注释给出）：
```
#sdy.op_sharding_rule<([i, j], [i, j])->([i, k], [i, k]) {i=2, j=2, k=4} need_replication={j, k}>
```
三步：
1. **两个操作数必须同分片** → 把 `%arg1` reshard 到 `[{"y"}, {}]`
2. 执行 `all_gather`
3. 结果分片与目标不符 → **再 reshard** 回去

**为什么有 `need_replication`**：`all_gather` 在 `all_gather_dim` 上把各设备的数据
**拼起来**，要求每台设备能看到**完整输入**（`j` 是输入的非聚合维、`k` 是输出新增的聚合维）。

**对比 `all_reduce`**：规则是 `([i,j],[i,j])->([i,j],[i,j])` ——
输出形状与输入**完全相同**，所以**没有** `need_replication`。

**一句话**：这三类算子都**不是透明的** —— 都有自己的分片约束，都需要显式 reshard 对齐。

## 练习

见第 6 幕。三道题分别考 gather 的四步、custom 规则的地位、集合通信的自身规则。

## 验收点

- [x] `check_ir_fidelity.py`：12 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 gather / custom_call / collective 各自的约束来源
