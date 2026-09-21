# L6-02 · `exec-stablehlo-collectives` — stablehlo 通信的执行

> 层：**L6 · 执行与解释器** ｜ 优先级：P1 ｜ 前置课：`L6-00`、`L6-01`、`L4-11`

## 学习目标

看完这一课，你应该能：

1. **说出与 L6-01 对应文件的异同**（TODOLIST 验收点）；
2. 说出 `manual_computation` 包裹在这里的作用（对应哪一课）；
3. 说出**期望值的三种写法**及其选择依据；
4. 说出"区域内形状"由什么决定。

## 覆盖的测试文件（5 个）

| 文件 | 通信算子 | 行数 |
|---|---|---|
| `executable_convert_global_to_local/stablehlo_all_to_all.mlir` | `all_to_all` | 60 |
| `executable_convert_global_to_local/stablehlo_all_reduce.mlir` | `all_reduce` | 51 |
| `executable_convert_global_to_local/stablehlo_all_gather.mlir` | `all_gather` | 48 |
| `executable_convert_global_to_local/stablehlo_collective_permute.mlir` | `collective_permute` | 45 |
| `executable_convert_global_to_local/stablehlo_reduce_scatter.mlir` | `reduce_scatter` | — |

网格是**单轴**的：`sdy.mesh @mesh_4 = <["x"=4]>` → `replica_groups` 都是 `[[0,1,2,3]]`。

## 场景（6 幕）

1. **★ 与 L6-01 的关键差异**
2. **★ 共同结构：`manual_computation` 包裹**
3. `all_reduce` 与 `all_gather`：期望值的两种写法
4. **★ `collective_permute`：循环置换**
5. `reduce_scatter` 与 5 个文件的族谱
6. 小结：两条路径，同一个结果

## 核心结论

### ★ 与 L6-01 的差异

| | L6-01（`sdy_*`） | **L6-02（本课）** |
|---|---|---|
| part1 写的 | `sdy.all_reduce` | **`manual_computation` + `stablehlo.all_reduce`** |
| 抽象层次 | **高层** | **低层** |
| `replica_groups` | 降级**生成** | **手写** |
| 对应课 | L5-02（降级） | **L4-11**（逐指令分区） |

**★ 两者语义等价** —— 因为 L5-02 的降级产物**就是** `stablehlo.*`。
**同一个 `all_reduce`，两条路径结果都是 `1111`** → 降级语义保持 ✓

### ★ `manual_computation` 包裹（L4-11 的形态）

- `manual_axes={"x"}` —— **冻结 `x` 轴**，区域内不再分片。
- 区域内**直接放 `stablehlo` 算子** —— 分片已「手动处理」。
- `in_shardings` 是**全局分片**；区域内 `%arg1` 是**局部形状**。

**一个可辨识的细节**：`channel_handle` 的 `type = 0`（手写），
而 L5-02 降级生成的是 `type = 1`。

### ★ 期望值的三种写法

| 算子 | 写法 | 为什么 |
|---|---|---|
| `all_reduce` | **硬编码 `1111`** | 求和结果最简单 |
| `all_gather` | **`concatenate` 现算** | 语义就是拼接 |
| `collective_permute` | **各设备的输入** | 收发对照表的直接体现 |

**现算的好处**：`concatenate` 就是 `all_gather` 的语义 —— **最不容易写错**。

### ★ `collective_permute` 的循环置换

`source_target_pairs = [[0,1], [1,2], [2,3], [3,0]]`：

| 设备 | 收到谁的 | 期望值 |
|---|---|---|
| 0 | 设备 **3** | `%s3` |
| 1 | 设备 **0** | `%s0` |
| 2 | 设备 **1** | `%s1` |
| 3 | 设备 **2** | `%s2` |

**读它的关键是注意方向**：`[0, 1]` 读作「设备 0 **发给**设备 1」
→ 所以**设备 1 收到 `%s0`**。

**对比 L5-02**：那里 `[1,4]`/`[4,1]` **成对出现**（交换）；
这里是**循环** → 说明 `source_target_pairs` **不要求成对**。

### 区域内形状由 `in_shardings` 决定

| `in_shardings` | 区域内类型 |
|---|---|
| `[{"x"}, {}]` | 局部 `4x8`（已分片） |
| `[{}, {}]` | 完整 `16x8`（全复制） |

`reduce_scatter` 用后者 —— 所以区域内是**完整数据**，然后「边归约边切分」。

## 验收点

- [x] `check_ir_fidelity.py`：30 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：5 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出与 L6-01 对应文件的异同
