# L6-04 · `exec-matmul-fft-iota` — 矩阵乘 / FFT / iota 的执行

> 层：**L6 · 执行与解释器** ｜ 优先级：P1 ｜ 前置课：`L6-00`、`L5-04`、`L5-05`、`L5-09`

## 学习目标

看完这一课，你应该能：

1. **解释 `dot_general_indivisible` 为何需要 padding**（TODOLIST 验收点）；
2. 说出 iota 的测试为什么**恰好能抓住漏掉偏移补偿的 bug**；
3. 说出 FFT 为什么**无需通信**；
4. 说出 `dot_general` 的收缩维分片为什么需要 `all_reduce`。

## 覆盖的测试文件（4 个）

| 文件 | 验证什么 | 对应课 |
|---|---|---|
| `executable_convert_global_to_local/stablehlo_iota.mlir` | **iota 的偏移补偿** | **L5-04** |
| `executable_convert_global_to_local/stablehlo_fft.mlir` | FFT **作用维未分片** | **L5-04** |
| `executable_convert_global_to_local/stablehlo_dot_general.mlir` | 收缩维分片 = **数据并行** | **L5-05** |
| `executable_partitioner_pipeline/stablehlo_dot_general_indivisible.mlir` | **不可整除补齐 3→4** | **L5-09** |

网格：`sdy.mesh @mesh_2 = <["x"=2]>`

## 场景（6 幕）

1. **★ 四个文件，验证三条规律**
2. **★ `iota`：验证 L5-04 的偏移补偿**
3. **★ `fft`：作用维未分片**
4. **★ `dot_general`：收缩维分片 = 数据并行**
5. **★★ `dot_general_indivisible`：验证 L5-09 的 3→4 补齐**
6. 小结：四课验证一条链

## 核心结论

### ★ iota：偏移补偿

`@parallel_iota` **无参数**，全局 `tensor<2xi32>` 沿 `{"x"}` 切 → 每台 1 个元素。

| | 设备 0 | 设备 1 | 拼接 |
|---|---|---|---|
| **有偏移补偿**（正确） | `[0]` | `[1]` | **`[0, 1]`** ✓ |
| **无偏移补偿**（错误） | `[0]` | `[0]` | **`[0, 0]`** ✗ |

**★ 这个测试恰好能抓住那个 bug** —— `[0,0]` 与 `[0,1]` 明显不同。

### ★ fft：作用维未分片

- **分片在第 0 维（批维）**；**FFT 作用在第 1 维**（`fft_length = 4`）。
- → **作用维未被分片** → 每台独立做 4 点 FFT → **无通信** ✓

**这正是 L5-04 的「作用维判据」。** 注释直接给出结论：
`each device performs a 4-point FFT`。

### ★ dot_general：数据并行

`lhs_contracting_dimensions = [1]`、`rhs_contracting_dimensions = [0]`
—— **两个收缩维都被切** → 产生**部分和** → `unreduced={"x"}` + `all_reduce`。

**注释直接标注了分片方案**：
```
// Device 0: lhs[:, 0:2], rhs[0:2, :]
// Device 1: lhs[:, 2:4], rhs[2:4, :]
```

### ★★ dot_general_indivisible：3→4 补齐

```mlir
// Contracting dimension is sharded and indivisible (padded 3->4).
```

- **收缩维大小 3**，沿 `y=2` 分片 → **除不尽** → **补到 4** ✓
- 结果类型仍是 `tensor<3x5xf32>` —— **补齐是内部的**，最后 `slice` 裁回。

**★ 这是 L5-09 那条规则最直接的证据**：
> L5-09 我从「补到**下一个能被轴整除的数**」出发，
> 给出验收点 `tensor<7x3x8>` 沿 `z=3` 分片 → 补到 `9`。
> **本课的测试文件用一个真实例子（3 → 4）证实了它。**

**两个 RUN 行**验证 `enable-halo-exchange` 的 `true`/`false` 结果一致（L4-09）。

### 三个可迁移的观察

1. **好的测试能区分正确与错误实现** —— iota 的例子
2. **上游测试的注释常直接点明意图** —— 值得先读（FFT 的注释）
3. **注释常直接标注分片方案** —— 读这类测试的最快入口（dot_general 的注释）

## 验收点

- [x] `check_ir_fidelity.py`：18 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：4 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能解释 `dot_general_indivisible` 为何需要 padding（3 沿 y=2 除不尽 → 补到 4）
