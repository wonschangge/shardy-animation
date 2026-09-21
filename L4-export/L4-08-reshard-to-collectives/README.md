# L4-08 · `reshard-to-collectives` — reshard 转集合通信（★ 核心）

> 层：**L4 · 导出流水线** ｜ 优先级：P0 ★ 核心 ｜ 前置课：`L1-06`、`L4-02`

## 学习目标

看完这一课，你应该能：

1. 背出**核心对照表**（分片变化 → 通信）；
2. 说出冗余 reshard 的**消除条件**，以及为什么**跨 mesh**、**设备序不同**也可能冗余；
3. 说出 `all_to_all` 为什么有时需要**先 `all_slice`**；
4. 区分 **replace / swap / reorder** 三类轴操作；
5. 说出 `keep-redundant-reshards` 选项的用途。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/reshard_to_collectives.mlir` | **960** | **98** |
| `transforms/export/test/reshard_to_collectives_keep_redundant_reshards_true.mlir` | 27 | 3 |

## 场景（6 幕）

1. **★ 核心对照表：分片变化 → 通信**
2. **★ 冗余 reshard 的消除**
3. 复杂情形：先 slice 再 all_to_all
4. 三类轴操作：replace / swap / reorder
5. 101 个用例的族谱与一个选项
6. 练习

## 核心结论

### ★ 核心对照表

| 分片变化 | 通信 | 参数含义 |
|---|---|---|
| **去掉**某维的轴 | `sdy.all_gather` | 要**聚合掉的轴的位置** |
| **加上**某维的轴 | `sdy.all_slice` | **目标分片** |
| 轴**从一个维移到**另一个维 | `sdy.all_to_all` | `[{"x"}: 0->2]`（哪个轴：从哪维到哪维） |
| 轴在**两个维之间交换** | `sdy.collective_permute` | —（前提：**尺寸相同**） |
| **前后分片相同** | **无（直接删除）** | — |

`all_gather` 与 `all_slice` **互逆**。

### ★ 冗余 reshard 的消除

- 前后分片一致 → 整条 reshard **被删掉**（`return %arg0`）。
- **跨 mesh 也可能冗余**：`@mesh2d_2x3` → `@mesh1d_6`，两者都是全复制。
- **设备序不同也可能冗余**：`@mesh2d` → `@mesh2d_non_iota`（`device_ids=[3,2,1,0]`）。
- **关键洞察**：**全复制状态下，网格的具体形状无关紧要** —— 每台设备都有完整副本。
  这**修正了 L4-07 的结论**：「设备序不同需要 reshard」在全复制状态下**不适用**。

### 复杂情形

- **`all_to_all` 要求源维与目标维的轴尺寸匹配** —— 不匹配时先 `all_slice` 调整。
  这一族 10 个用例（含"两个 all_to_all"、"再叠 all_gather"等组合）。
- **9 个「不能 slice」的用例**锁定失败条件：输出维已有分片、目标维已有分片、
  轴序不对、轴不连续、尺寸太小、尺寸不可整除。
  → **「先 slice」不是万能的**；负面用例与正面用例**同等重要**。

### 三类轴操作

| 操作 | 含义 | 特点 |
|---|---|---|
| **replace** | 一个轴被**另一个取代** | 26+ 用例（尺寸关系 × 维度关系） |
| **swap** | 两个维的轴**互换** | **跨维**操作 |
| **reorder** | 同一个维内轴的**顺序**变化 | 可以是**同维内**的 |

记忆法：**`swap` 跨维、`reorder` 同维内、`replace` 是取代**。

另有 **`gcd > 1`** 的 4 个用例：两轴大小的最大公约数 > 1 时可**部分匹配**
（用子轴切出公共部分），不必整体 slice —— 通信量更小。

### 选项

`keep-redundant-reshards=true`：**不删除**冗余 reshard，便于调试时看清全貌。
与 L2-03 的 `keep-sharding-rules`、L4-05 的 `mark-partial-result` 同类（**观测性**选项）。

## 练习

见第 6 幕。三道题分别考对照表、冗余消除、复杂组合。

## 验收点

- [x] `check_ir_fidelity.py`：24 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：给定 reshard 前后分片，能写出对应的 collective 序列
