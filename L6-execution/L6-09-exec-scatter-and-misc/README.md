# L6-09 · `exec-scatter-and-misc` — scatter 与杂项（L6 收官）

> 层：**L6 · 执行与解释器** ｜ 优先级：P1 ｜ 前置课：`L6-00`、`L5-08`、`L5-04`

## 学习目标

看完这一课，你应该能：

1. **解释 scatter 隐式维与显式维分片的区别**（TODOLIST 验收点）；
2. 说出 `replicated_bounds` 与 `shard_implicit_dim` 的**用例意图差异**；
3. 说出四个 misc 算子的**共同点**；
4. 说出 `sort` 的排序维若被分片会怎样。

## 覆盖的测试文件（7 个，全在 `executable_convert_global_to_local/`）

| 族 | 文件 | 行数 |
|---|---|---|
| **scatter** | `stablehlo_scatter_shard_indexed_inserted_dim.mlir` | 77 |
| **scatter** | `stablehlo_scatter_shard_implicit_dim.mlir` | 76 |
| **scatter** | `stablehlo_scatter_replicated_bounds.mlir` | 70 |
| **misc** | `stablehlo_select_and_scatter.mlir` | 70 |
| **misc** | `stablehlo_reduce_window.mlir` | 51 |
| **misc** | `stablehlo_concatenate.mlir` | 49 |
| **misc** | `stablehlo_sort.mlir` | 43 |

## 场景（5 幕）

1. **★ scatter 的三种分片形态**
2. **★ 隐式维 vs 显式维分片（验收点）**
3. `replicated_bounds`：边界全复制
4. **★ 四个 misc 算子：作用维都未分片**
5. **L6 收官：十课的主线**

## 核心结论

### ★ scatter 的三种形态

三个文件的 `scatter_dimension_numbers` **几乎相同**，区别在**谁被分片**：

| 文件 | `%arg0`（bounds） | `%arg1`（索引） | `%arg2`（更新值） |
|---|---|---|---|
| `shard_implicit_dim` | **全复制** | 切 `{"x"}` | 切 `{"x"}` |
| `shard_indexed_inserted_dim` | **切 `{"x"}`** | **无分片** | **无分片** |
| `replicated_bounds` | **全复制** | 切 `{"x"}` | 切 `{"x"}` |

### ★ 验收点：隐式维 vs 显式维

**隐式维**（`shard_implicit_dim`）：`%arg1` 的第 0 维是**索引的个数** ——
它在 `scatter_dimension_numbers` 里**没有对应的 operand 维**
（`scatter_dims_to_operand_dims = [0]` 指的是 `%arg1` 的**第 1 维**映射到 `%arg0` 的第 0 维）
→ 所以第 0 维是**隐式的批维**。**各写各的位置**。

**显式维**（`shard_indexed_inserted_dim`）：`%arg0` 的**行维被切开** ——
注释直接点明 `// Row dimension is collapsed/inserted`。
索引可能指向**别的设备持有的行** → **需要索引重映射**（L5-08 的八步）。

**★ 这是 L5-08 索引重映射在 `scatter` 上的完整执行验证。**

### `replicated_bounds`

分片位置与 `shard_implicit_dim` **相同**，但**用例意图不同** ——
强调「**bounds 全复制**」这个前提对**合并策略**的影响。

**scatter 是写操作** → 多台可能写同一位置 → 用 `update_computation` 合并（而非 `all_reduce`）。

### ★ 四个 misc 算子的共同点

| 文件 | 作用维 | 分片维 | 通信 |
|---|---|---|---|
| `concatenate` | **拼接维** `dim = 1` | 第 0 维 | **无** |
| `sort` | **排序维** `dimension = 1` | 第 0 维 | **无** |
| `reduce_window` | **窗口维** `[1, 2]` | 第 0 维 | **无** |
| `select_and_scatter` | **窗口维** `[1, 2]` | 第 0 维 | **无** |

**作用维都在第 1 维、分片都在第 0 维** → 全部无通信。
这是 **L5-04 作用维判据**在更多算子上的验证。

**`sort` 的特殊性**：若**排序维被分片**，排序需要**看到整行**才能确定顺序
→ 跨设备时要通信（类似 `reverse` 的情形）。

## L6 收官：三条主线

1. **可执行测试 = 分片版 vs 串行版的数值对比**（L6-00）
2. **集合通信的两条路径**：SDY 算子降级 vs 手写 `stablehlo`（L6-01/02）
3. **各算子族的执行验证**：归约方向分片 → 通信（03/04/05）/
   `pad` 是 halo exchange 的载体（06）/ `reshape` 触发分片重排（07）/
   `slice`·`reverse` 看是否改变数据归属（08）/ `scatter` 看分片落在哪（09）

**L6 的两个实测发现**：
- **gather 的 mask 填充值是归约的单位元**（L6-05）—— 修正了 L5-08 的「填零」表述
- **pad 的 uniform 判据**是 `pLow + pHigh` 与 `pInt` 的关系（L6-06）

## 验收点

- [x] `check_ir_fidelity.py`：21 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：7 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能解释隐式维与显式维分片的区别
