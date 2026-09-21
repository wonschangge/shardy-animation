# L4-10 · `optimize-collectives` — 通信优化

> 层：**L4 · 导出流水线** ｜ 优先级：P2 ｜ 前置课：`L4-08`、`L4-09`

## 学习目标

看完这一课，你应该能：

1. 说出这个优化**做了什么**、省下了什么；
2. **指出哪条 `permute` 被消除以及为什么安全**（TODOLIST 验收点）；
3. 区分**全散开**与**部分散开**；
4. 说出优化**不是无条件的**——有哪些边界情形。

## 覆盖的测试文件

| 文件 | 行数 | 用例数 |
|---|---|---|
| `transforms/export/test/optimize_collectives/all_to_all_fully_scattered.mlir` | 133 | 7 |
| `transforms/export/test/optimize_collectives/all_to_all_partially_scattered.mlir` | 98 | 6 |

RUN 行：`-sdy-optimize-collectives`

## 场景（6 幕）

1. **★ 优化内容：删掉冗余的 `permute`**
2. **★ 为什么删 `permute` 是安全的**
3. 两类散开：`full` vs `partial`
4. 覆盖的边界情形
5. 13 个用例的族谱与小结
6. 练习

## 核心结论

### 优化内容

```
优化前：collective_permute → all_to_all × 2          （3 步，含一次真实通信）
优化后：reshape → all_to_all × 2 → reshape           （4 步，但 reshape 无通信）
```

**模式**：`collective_permute`（删除）→ `reshape`（拆复合轴）→
`all_to_all × N`（按轴搬）→ `reshape`（合回去）。

**省下的是一次跨设备通信** —— 整个流程里最贵的一环。

### ★ 为什么安全

> **`all_to_all` 按【轴名】搬运，所以轴在源维内的顺序不影响结果。**

推理链：
1. `all_to_all [{"x"}: 0->2]` 指定的是**轴名**，不是"第几个"
2. 轴在源维内的**顺序**不影响搬运结果
3. `permute` 做的事只是把 `{"x","y"}` 换成 `{"y","x"}` —— **纯顺序调整**
4. → 对后续 `all_to_all` **没有影响** → **可安全删除**

### 两类散开

| 类别 | 文件 | 用例数 | 定义 |
|---|---|---|---|
| **全散开** | `all_to_all_fully_scattered` | 7 | 复合轴里**所有**轴都被搬走 |
| **部分散开** | `all_to_all_partially_scattered` | 6 | **只有一个**轴被搬走，另一个留在原维 |

**一个反直觉点**：部分散开时**仍然有两条 `all_to_all`** ——
因为 `reshape` 拆开复合轴后，**留下的轴也换了位置**（到了新产生的维），还要搬回去。
所以"部分散开"并不意味着"通信更少"，只是**散开的轴更少**。

### 边界情形（6 类）

| 边界 | 为什么可能让优化失效 |
|---|---|
| **子轴** | 尺寸可能不可整除 |
| **非 major 切分维** | `reshape` 语义与"哪个维是 major"有关 |
| **未触及的轴** | 优化不能误伤不需要动的轴 |
| **下游还有通信** | 不能把别的 `all_to_all` 一起误删 |
| **循环置换** | `x->y->z->x` 最容易出错 |
| **搬到同一目标维** | 两个轴到了同一个维，**顺序就重要了** |

→ 证明优化是**有条件的**，不是"见到 permute + all_to_all 就删"。

### 与 L4-08 冗余消除的对比

| | 消除对象 | 为什么可以删 |
|---|---|---|
| **L4-08** | 前后分片相同的 `reshard` | 本来就不产生通信 |
| **L4-10** | `all_to_all` 前的 `permute` | 通信可被 `reshape` 吸收 |

## 练习

见第 6 幕。三道题分别考优化内容、安全性理由、边界条件。

## 验收点

- [x] `check_ir_fidelity.py`：7 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：2 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能指出哪条 permute 被消除以及为什么安全
