<!-- sdy-coverage
ir/op_interface.td
ir/constants.h
-->

# L7-04 · dialect-agnostic-integration — 源定义

**全部 75 课的最后一课**：如何为**自己的方言**接入 Shardy。

> **说明**：本课引用的是**接口定义**（`.td` / `.h`）而非测试 IR ——
> 它们是"怎么扩展 Shardy"的**权威定义**。
> 这两个文件**不在** `transforms/*/test/` 下，**不计入 241 个测试 IR**。

| 文件 | 提供什么 |
|---|---|
| `ir/op_interface.td` | **三个接口**的定义 |
| `ir/constants.h` | **常量拆分**相关的辅助 |

---

## 一、★ 三个接口

`op_interface.td` 定义了三个接口：

| 接口 | 行号 | 作用 |
|---|---|---|
| `Sdy_ShardableDataFlowOpInterface` | 27 | **数据流边** —— 让分片穿过算子传播 |
| `Sdy_ShardingRuleOpInterface` | 232 | **自定义 sharding rule** |
| `Sdy_CollectiveOpInterface` | 267 | **集合通信算子** |

**接入 Shardy 就是实现其中的一个或多个。**

---

## 二、★ `ShardingRuleOpInterface`：定义自己的分片规则

```cpp
def Sdy_ShardingRuleOpInterface : OpInterface<"ShardingRuleOpInterface"> {
  let description = [{
    An op interface that allows the op to define its own sharding rule.
    A sharding rule specifies how an operation can be partitioned according to
    various properties on the op - any attributes, the shape of operands,
    the shape of the results, etc. See `OpShardingRuleAttr` for more
    details.
  }];
  let cppNamespace = "::mlir::sdy";
    let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the sharding rule of the op.
      }],
      /*retType=*/"mlir::sdy::OpShardingRuleAttr",
      /*methodName=*/"getShardingRule"
    >,
    InterfaceMethod<
      /*desc=*/[{
```

**读法**（**本课的核心**）：
- **"allows the op to define its own sharding rule"** ——
  算子可以**自己声明**"我能怎么被切分"。
- 规则的依据：**算子的属性、操作数的形状、结果的形状**等。
- **核心方法**：**`getShardingRule()`** —— 返回一个 `OpShardingRuleAttr`。

**★ 回顾 L2-10**：那里讲 `op_sharding_rule` 的**语法** ——
```
#sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
```
**本课讲的是"怎么生成这条规则"** —— 实现这个接口。

**★ 两种生成方式**：
| 方式 | 适用 |
|---|---|
| **注册规则**（`-sdy-populate-op-sharding-rules`） | 用**声明式**的规则表 |
| **实现接口**（本课） | 规则的**逻辑复杂**、依赖运行时信息 |

**回顾 L7-03**：那里讲的警告
`custom call @unknown_custom_op is unknown to SDY sharding rule registry` ——
**就是"既没注册、也没实现接口"的结果**。

---

## 三、★ `ShardableDataFlowOpInterface`：数据流边

```cpp
def Sdy_ShardableDataFlowOpInterface : OpInterface<"ShardableDataFlowOpInterface"> {
  let description = [{
    An op interface that allows shardy to propagate shardings through data flow
    edges of ops that extend this interface.

    A data flow edge of some op X defines a bridge between a set of sources
    (each is either an operand of X or an operand of X's block terminator) and
    a set of targets (each is either a result of X or a block argument of X),
    such that all sources and targets should be sharded in the same way.
    An op can have multiple data flow edges that are orthogonal to one another.

    An owner is a user specified target of the data flow edge used by shardy's
    propagation. The user can choose it arbitrarily but it needs to be static.
```

**读法**（**这段定义非常关键**）：
- **"propagate shardings through data flow edges"** ——
  让分片**穿过算子**传播。
- **数据流边（data flow edge）的定义**：
  - 一端是 **sources**：算子 X 的**操作数**或**块终结符的操作数**
  - 另一端是 **targets**：算子 X 的**结果**或**块参数**
  - **要求**：所有 sources 和 targets **必须用同样的方式分片**
- **一个算子可以有多个正交的数据流边**。
- **owner**：用户指定的 target ——
  **"can choose it arbitrarily but it needs to be static"**（可以任选，但必须是静态的）。

**★ 为什么需要这个接口**：
> 回顾 **L2-07**：`data-flow edges` 描述"分片怎么从一个算子流到另一个算子"。
> 对于**区域算子**（`while`/`case`，**L7-03** 讲的 40 个用例），
> 数据流要**穿过 region 边界** —— 通用规则**推导不出来**，
> 必须由算子**自己告诉 Shardy**"哪些东西应该分片一致"。

**★ `sources` / `targets` 的四种组合**：

| 端 | 可以是 |
|---|---|
| **source** | X 的操作数 / X 的**块终结符的操作数** |
| **target** | X 的结果 / X 的**块参数** |

**"块终结符的操作数"与"块参数"** 正是**区域算子**特有的 ——
这就是为什么区域算子需要这个接口。

---

## 四、`CollectiveOpInterface`：集合通信算子

```cpp
def Sdy_CollectiveOpInterface : OpInterface<"CollectiveOpInterface"> {
  let description = [{
    Interface for all collective ops. Encapsulates common get/set for
    outSharding attribute.

    **Constraints:**
    - Operand must have a sharding or `allowMissingInputSharding()` returns
      true.
    - `out_sharding` is valid w.r.t the corresponding type.
    - Operand and result sharding must have the same mesh if
      `allowDifferentMeshes()` returns false.
    - Same rank for the operand and result sharding.
  }];
  let cppNamespace = "::mlir::sdy";
```

**读法**：
- **"Encapsulates common get/set for `outSharding` attribute"** ——
  统一管理 `out_sharding`（**L5-02** 反复强调的"`out_sharding` 是派生的"）。
- **四条约束**（很实用）：
  1. 操作数**必须有分片**（除非 `allowMissingInputSharding()` 返回 true）
  2. `out_sharding` 必须与类型**匹配**
  3. 操作数与结果的 mesh **必须相同**（除非 `allowDifferentMeshes()`）
  4. 操作数与结果的分片**秩必须相同**

**★ 回顾 L5-02**：那里看到 8 种集合通信算子的 `out_sharding` 语法 ——
**本课看到它背后的接口约束**。

---

## 五、★ Elementwise trait 与常量拆分

**除了接口，还有两个"轻量"的接入方式**：

### Elementwise trait

**逐元素算子**（如 `add`、`negate`）的分片规则**完全一样**：
**所有操作数与结果用同样的分片**。

**★ 为什么可以用 trait 而不是接口**：
> 逐元素算子的规则**不需要任何计算** —— 它是**固定的**。
> 用 **trait**（编译期标记）比实现接口（运行期调用）**更轻**。

**回顾 L5-04**：那里讲 `negate` 是"逐元素算子，分片后无需通信" ——
**这正是 Elementwise trait 的效果**。

### 常量拆分（`constants.h`）

`constants.h` 里多处提到 `ShardableDataFlowOpInterface`：

```cpp
// `ShardableDataFlowOpInterface` op block arguments.
// `ShardableDataFlowOpInterface` op results.
```

**★ 常量拆分的含义**（回顾 **L5-01 / L5-03**）：
> 一个**全局常量**（如 `dense<[[1,2],[3,4]]>`）分片后，
> 每台设备需要**不同的切片**。
> **"常量拆分"就是把常量按分片拆成每台设备的那一份。**

**★ 两种常量**（L5-03 讲过）：
| 常量 | 拆分方式 |
|---|---|
| **splat**（`dense<1.0>`） | **不需要拆分**（内容处处相同） |
| **dense**（元素各异） | 用 `replica_id` + 查找表 + 切片 |

**★ 与数据流边的关系**：
常量拆分**也是一种数据流边** ——
常量的结果与"使用它的算子"之间，分片要**一致**。

---

## 六、★ 接入步骤与验收点

**验收点：能写出一条自定义算子的 sharding rule 并被传播正确使用。**

### 三步接入

| 步 | 做什么 | 用什么 |
|---|---|---|
| **①** | 让算子**能定义分片规则** | 实现 `ShardingRuleOpInterface` 或**注册规则** |
| **②** | 让分片**能穿过算子** | 实现 `ShardableDataFlowOpInterface`（区域算子必需） |
| **③** | 若是**逐元素**算子 | 加 **Elementwise trait**（最轻） |

### 一个最小例子（sharding rule 的写法）

回顾 **L2-10** 的语法，为一个假想的 `my_op` 写规则：

```text
#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=8}>
```

**逐项读**：
- `([i, j], [j, k])->([i, k])` —— 两个操作数 `[i,j]`/`[j,k]`，结果 `[i,k]`。
- `{i=8, j=16, k=8}` —— 因子的**大小**。
- **`j` 出现在两个操作数里但不在结果里** → **`j` 是归约因子**（**L2-10** 的规则）。
- → 如果 `j` 被分片，就需要 **`all_reduce`**（**L5-05** 的判据）。

### 验证接入是否正确

| 检查 | 命令 |
|---|---|
| **规则有没有生效** | `sdy_opt -sdy-populate-op-sharding-rules -verify-diagnostics`（**L7-03**） |
| **传播对不对** | `sdy_opt -sdy-propagation-pipeline` 看结果 |
| **区域算子** | 参考 `data_flow_edges` 的 40 个用例（**L7-03**） |
| **数值对不对** | 写一个**可执行测试**（**L6-00** 的三要素） |

**★ 最后一条最重要**：
> 规则写得对不对，**最终要靠数值验证**（**L6-00** 讲的"分片版 vs 串行版"）。
> 这与 L6 全层的做法一致。

---

## 七、★ 全部 75 课的回顾

**这是最后一课。把七层串起来**：

| 层 | 主题 | 核心问题 |
|---|---|---|
| **L1** | SDY 方言基础 | 分片**怎么表达**？ |
| **L2** | 分片传播 | 分片**怎么流动**？ |
| **L3** | 导入 | 别的方言**怎么进来**？ |
| **L4** | 导出（reshard） | 分片**怎么变成通信**？ |
| **L5** | 导出（降级） | 全局张量**怎么变成局部**？ |
| **L6** | 执行与解释器 | 分片代码**怎么跑起来**？ |
| **L7** | 综合实战 | **怎么用、怎么排错、怎么扩展**？ |

**★ 三条贯穿全课的主线**：

1. **只有归约方向上的分片需要通信** ——
   L5-05（收缩维）/ L5-06（归约因子）/ L5-07（归约维）/ L5-08（collapsed 维）/ L6 全层验证。
2. **位置相关的算子要补偿位置** ——
   L5-04（iota 偏移）/ L5-08（索引重映射）/ L6-05（填充值 = 单位元）。
3. **不可整除时让通信发生在可整除的形状上** ——
   L5-09（pad 补齐 + slice 裁回）/ L6-06（`indivisible` 的 3→4）。

**★ 本课的位置**：
> **L7-04 是"向外"的一课** ——
> 前面 74 课讲"Shardy 怎么工作"，这一课讲"**怎么让 Shardy 为你的方言工作**"。
>
> 三个接口 + 一个 trait，就是全部的扩展点。

**一句话总结**：
> **接入 Shardy 就是实现三个接口之一** ——
> `ShardingRuleOpInterface`（定义规则）、
> `ShardableDataFlowOpInterface`（穿过算子）、
> `CollectiveOpInterface`（通信算子）；
> 逐元素算子加 **Elementwise trait** 即可，常量走**常量拆分**。
