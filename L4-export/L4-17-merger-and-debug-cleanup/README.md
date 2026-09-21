# L4-17 · `merger-and-debug-cleanup` — 合并与调试清理（L4 收官）

> 层：**L4 · 导出流水线** ｜ 优先级：P2 ｜ 前置课：`L3-02`、`L4-15`、`L4-16`

## 学习目标

看完这一课，你应该能：

1. 说出 L3-02 **拆开**常量与本课**合并**常量为什么**两次都对**；
2. 说出 `propagate_to_func_results` **补全的是什么**、为什么需要；
3. 说出 `propagation_edges` **为什么可以删**；
4. 回顾 L4 全层 17 课的**主线**。

## 覆盖的测试文件（3 个 / 483 行 / 35 用例）

| pass | 文件 | 行数 | 用例数 |
|---|---|---|---|
| `--sdy-constant-or-scalar-merger` | `constant_or_scalar_merger.mlir` | 182 | 13 |
| `-sdy-propagate-to-func-results` | `propagate_to_func_results.mlir` | 189 | 20 |
| `-sdy-remove-propagation-debug-info` | `remove_propagation_debug_info.mlir` | 112 | 2 |

## 场景（6 幕）

1. L4 最后一课：优化、补全、清理
2. **★ 常量合并：与 L3-02 的镜像关系**
3. 补全函数结果分片 & 删调试信息
4. **L4 收官：17 课的主线**
5. 3 个 pass / 35 个用例的族谱
6. 练习

## 核心结论

### 三个 pass 的三种类型

| pass | 做什么 | 类型 |
|---|---|---|
| `constant_or_scalar_merger` | 合并重复常量 | **优化**（IR 变小） |
| `propagate_to_func_results` | 补全函数结果分片 | **补全**（信息完整） |
| `remove_propagation_debug_info` | 删 `sdy.propagation_edges` | **清理**（去噪声） |

### ★ 常量合并：与 L3-02 的镜像

| 课 | 方向 | 为什么 |
|---|---|---|
| **L3-02** | N 个使用 → **N 份**（拆分） | 避免传播期产生**假依赖** |
| **L4-17** | 相同内容 → **一份**（合并） | 传播已结束，假依赖不再是问题 |

**为什么两次都对**：两个阶段的**目标不同**。
导入期需要「传播自由度」；导出期只需要「IR 更小、编译更快」。

**L4 的三类镜像对**：
- L3-02 拆分 ↔ **L4-17 合并**
- L3-06 内联 ↔ L4-12 outline
- L3-07 提升 ↔ L4-15 内联

→ 共同的道理：**导入期为了「传播自由度」做的事，导出期都要还原。**

### `propagate_to_func_results`

- **补全的是函数结果签名上的分片**。
- 情形：终止符值**有**分片，但函数结果签名**没有** → 传播过去。
- **为什么需要**：函数结果的分片必须**反映实际返回值的分片**。
- **与 L4-16 的两步配合**：先把**边**的分片下沉到值上，再把**值**的分片传播到签名上。

### `remove_propagation_debug_info`

- `sdy.propagation_edges` **不是语义信息** —— 只描述「分片是怎么传播过来的」。
- 只在**调试**时有用；保留会让 IR 变大、干扰后续处理。
- **同类观测性选项**：L2-03 `keep-sharding-rules`、L4-05 `mark-partial-result`、
  L4-08 `keep-redundant-reshards`。

### L4 全层主线

```
传播后的 IR
  → 插入 reshard（L4-02～07，按算子族展开）
  → 转成集合通信（L4-08～10）
  → 处理特殊结构（L4-11～14）
  → 收尾清理（L4-15～17）
= 交给后端的 IR
```

**四条判据**（L4-02～07）：① 冲突在哪两侧 ② 有无 `reduction` 因子
③ 能否靠归约合并 ④ 是否跨边界。

**核心对照表**（L4-08）：去掉轴→`all_gather`；加上轴→`all_slice`；
轴跨维移动→`all_to_all`；轴互换→`collective_permute`；无变化→删除。

## 练习

见第 6 幕。三道题分别考常量合并、函数结果补全、调试清理。

## 验收点

- [x] `check_ir_fidelity.py`：18 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出为什么拆分与合并两次都对
