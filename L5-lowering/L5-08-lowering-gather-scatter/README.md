# L5-08 · `lowering-gather-scatter` — 访存类降级

> 层：**L5 · 传播与降级** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L3-11`、`L5-07`

## 学习目标

看完这一课，你应该能：

1. 说出访存类算子**为什么最难降级**；
2. **背出索引重映射的八个步骤**；
3. 说出 `gather` 与 `scatter` 的**对称性**与差异；
4. 说出为什么"不属于我的位置填零"是安全的。

## 覆盖的测试文件（3 个 / 956 行 / 22 用例）

| 文件 | 行数 | 用例数 |
|---|---|---|
| `convert_global_to_local/stablehlo_gather.mlir` | 463 | 11 |
| `convert_global_to_local/stablehlo_scatter.mlir` | 416 | 9 |
| `convert_global_to_local/stablehlo_select_and_scatter.mlir` | 77 | 2 |

## 场景（5 幕）

1. **★ 核心问题：索引重映射**
2. **★ 索引重映射的八个步骤**
3. `gather` 的 11 个用例
4. `scatter`：被索引的维分片
5. `select_and_scatter` 与统一规律

## 核心结论

### ★ 为什么访存类最难

> `gather` / `scatter` 的索引是**全局坐标**，但每台设备只持有**局部数据**
> → 索引必须**转换到本地坐标系**。

| 课 | 问题 | 解法 |
|---|---|---|
| L5-04 `iota` | **序号**起点不同 | 本地 iota + 加偏移 |
| L5-07 `reduce` | **部分结果**需合并 | `all_reduce` / 收齐再算 |
| **L5-08** | **索引**指向别的设备 | **索引重映射 + mask** |

**最复杂的原因**：不仅要把索引**平移**，还要处理「**这个索引根本不属于我**」
→ 需要 **mask + select 填零**。

### ★ 索引重映射的八个步骤

| 步 | 操作 | 作用 |
|---|---|---|
| ① | `clamp` | 索引夹紧到全局范围 |
| ② | `partition_id` + 查表 | 得到本设备**偏移** |
| ③ | `broadcast_in_dim` | 标量 → 张量 |
| ④ | `subtract` | **索引转换到本地坐标系** |
| ⑤ | `compare` + `and` | 生成 **mask**（是否属于我） |
| ⑥ | `gather` | 用**本地索引** |
| ⑦ | `select` + 0 | **不属于我的填零** |
| ⑧ | `all_reduce` | 合并部分结果 |

**为什么填零安全**：`all_reduce` 用**加法** —— 填 0 不影响和。

### `gather` 的 11 个用例

关键区分：**归约维是否被 `collapsed`**。
- **被 collapsed + 分片** → 索引重映射 + `all_reduce`
- **未被 collapsed** → 分片在输出维 → **无通信**

### `scatter` 与 `gather` 的对称性

| | `gather` | `scatter` |
|---|---|---|
| 问题维 | **归约维**（collapsed） | **被索引的维** |
| 索引处理 | 平移 + mask | 平移 + mask |
| 通信 | `all_reduce` | 可能不需要（写不同位置时互不冲突） |

**`scatter` 多的复杂度**：它有**更新值**，所以要区分 `inserted` / `non_inserted`。

### `select_and_scatter`

与 `reduce_window` **完全同构**（它是反向操作）—— 判据都是**窗口是否跨设备**。

## 验收点

- [x] `check_ir_fidelity.py`：26 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出索引重映射的八步、以及为什么填零是安全的
