# L7-04 · `dialect-agnostic-integration` — 跨方言集成（P1 · **全部 75 课的最后一课**）

> 层：**L7 · 综合实战** ｜ 优先级：P1 ｜ 前置课：`L2-10`、`L2-07`、`L5-02`、`L7-03`

## 学习目标

看完这一课，你应该能：

1. **写出一条自定义算子的 sharding rule 并被传播正确使用**（TODOLIST 验收点）；
2. 说出**三个接口**各自的作用与适用场景；
3. 说出**为什么区域算子需要** `ShardableDataFlowOpInterface`；
4. 说出 `CollectiveOpInterface` 的**四条约束**。

## 引用说明

本课引用的是**接口定义**（`.td` / `.h`）而非测试 IR：

| 文件 | 提供什么 |
|---|---|
| `ir/op_interface.td` | **三个接口**的定义 |
| `ir/constants.h` | **常量拆分**相关的辅助 |

> 这两个文件**不在** `transforms/*/test/` 下，**不计入 241 个测试 IR**。

## 场景（6 幕）

1. **★ 最后一课：三个接口就是全部扩展点**
2. **★ `ShardingRuleOpInterface`：定义自己的规则**
3. **★ `ShardableDataFlowOpInterface`：数据流边**
4. `CollectiveOpInterface` 与 `Elementwise trait`
5. **★ 接入步骤与验收点**
6. **★ 全部 75 课回顾与三条主线**

## 核心结论

### ★ 三个接口 + 一个 trait

| 接口 | 作用 | 何时用 |
|---|---|---|
| **`ShardingRuleOpInterface`** | 定义**分片规则** | 普通计算算子（或**注册规则**） |
| **`ShardableDataFlowOpInterface`** | **数据流边** —— 分片穿过算子 | **区域算子必需** |
| **`CollectiveOpInterface`** | **通信算子** —— 统一 `out_sharding` | 集合通信算子 |
| **Elementwise trait** | 逐元素算子 | **最轻**的方式 |

### `ShardingRuleOpInterface`

```cpp
def Sdy_ShardingRuleOpInterface : OpInterface<"ShardingRuleOpInterface"> {
  let description = [{
    An op interface that allows the op to define its own sharding rule.
    A sharding rule specifies how an operation can be partitioned according to
    various properties on the op - any attributes, the shape of operands,
    the shape of the results, etc. See `OpShardingRuleAttr` for more
    details.
  }];
```

**核心方法**：**`getShardingRule()`** → 返回 `OpShardingRuleAttr`。

**两种生成方式**：
- **注册规则**（`-sdy-populate-op-sharding-rules`）—— 用**声明式**规则表
- **实现接口**（本课）—— 规则**逻辑复杂**、依赖运行时信息

**★ 本课是 L7-03 那条警告的解决方案**：
`custom call ... is unknown to SDY sharding rule registry` = **既没注册、也没实现接口**。

### ★ `ShardableDataFlowOpInterface`

```cpp
    A data flow edge of some op X defines a bridge between a set of sources
    (each is either an operand of X or an operand of X's block terminator) and
    a set of targets (each is either a result of X or a block argument of X),
    such that all sources and targets should be sharded in the same way.
    An op can have multiple data flow edges that are orthogonal to one another.

    An owner is a user specified target of the data flow edge used by shardy's
    propagation. The user can choose it arbitrarily but it needs to be static.
```

| 端 | 可以是 |
|---|---|
| **source** | X 的操作数 / X 的**块终结符的操作数** |
| **target** | X 的结果 / X 的**块参数** |

**核心约束**：一条边上的所有 sources 与 targets **必须分片方式相同**。

**★ 为什么区域算子需要它**：数据流要**穿过 region 边界**，
通用规则**推导不出来** —— 必须由算子**自己声明**「哪些东西应该分片一致」。
（对应 **L7-03** 的 40 个用例。）

### `CollectiveOpInterface` 的四条约束

1. 操作数**必须有分片**（除非 `allowMissingInputSharding()`）
2. `out_sharding` 必须与类型**匹配**
3. 操作数与结果的 mesh **必须相同**（除非 `allowDifferentMeshes()`）
4. 操作数与结果的分片**秩必须相同**

### ★ 接入步骤

| 步 | 做什么 | 用什么 |
|---|---|---|
| **①** | 让算子**能定义规则** | 接口 或 注册规则 |
| **②** | 让分片**能穿过算子** | `ShardableDataFlowOpInterface`（区域算子） |
| **③** | 若是**逐元素**算子 | **Elementwise trait** |
| **④** | **验证数值** | 写**可执行测试**（L6-00 的三要素） |

### 一个最小例子

```text
#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=8}>
```

**逐项读**：`j` 出现在**两个操作数**里但**不在结果**里
→ **`j` 是归约因子**（L2-10）→ 若 `j` 被分片就需要 **`all_reduce`**（L5-05）。

### ★ 全部 75 课的回顾

| 层 | 核心问题 |
|---|---|
| **L1** | 分片**怎么表达**？ |
| **L2** | 分片**怎么流动**？ |
| **L3** | 别的方言**怎么进来**？ |
| **L4** | 分片**怎么变成通信**？ |
| **L5** | 全局张量**怎么变成局部**？ |
| **L6** | 分片代码**怎么跑起来**？ |
| **L7** | **怎么用、怎么排错、怎么扩展**？ |

**★ 三条贯穿全课的主线**：
1. **只有归约方向上的分片需要通信**（L5-05～08 + L6 全层验证）
2. **位置相关的算子要补偿位置**（L5-04 / L5-08 / L6-05）
3. **不可整除时让通信发生在可整除的形状上**（L5-09 / L6-06）

## 验收点

- [x] `check_ir_fidelity.py`：0 个 IR 块（本课引用接口定义，非 IR）
- [x] `check_coverage.py`：**241/241 = 100%**，B 组全部通过
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能写出 `#sdy.op_sharding_rule<...>` 并说明哪个是归约因子
