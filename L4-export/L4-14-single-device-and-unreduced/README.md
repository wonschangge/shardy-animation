# L4-14 · `single-device-and-unreduced` — 单设备与未归约

> 层：**L4 · 导出流水线** ｜ 优先级：P1 ｜ 前置课：`L1-01`、`L4-04`、`L4-05`

## 学习目标

看完这一课，你应该能：

1. 说出单设备分片在导出期**怎么处理**；
2. **说出 `unreduced` 轴从哪来、必须在哪被消除**（TODOLIST 验收点）；
3. 说出 **blessed operation** 的含义与作用；
4. 说出 CMV1 移除 `all_gather` 的**判断依据**。

## 覆盖的测试文件（3 个 / 618 行）

| 文件 | 行数 | 用例数 | pass |
|---|---|---|---|
| `transforms/export/test/resolve_single_device_sharding.mlir` | 186 | 5 | `-sdy-resolve-single-device-sharding` |
| `transforms/export/test/verify_unreduced_axes.mlir` | 302 | 0（全 `expected-error`） | `-sdy-verify-unreduced-axes` |
| `transforms/export/test/remove_ag_rs_for_cmv1.mlir` | 130 | 10 | `-sdy-remove-all-gather-reduce-scatter-for-cmv1` |

## 场景（6 幕）

1. 三类特殊情况
2. **★ 单设备分片 → `if` 守卫**
3. **★ `blessed operation`：未归约轴不能悄悄丢**
4. `remove_ag_rs_for_cmv1`：后端能做的就移除
5. 三个 pass 的共同点与小结
6. 练习

## 核心结论

### ① 单设备分片 → `if` 守卫（五步）

| 步 | 操作 |
|---|---|
| ① | 输入 **reshard 成全复制**（所有设备都要执行这条 `if`，都需要数据） |
| ② | `stablehlo.partition_id` 取当前设备号 |
| ③ | `stablehlo.convert` + `compare EQ` 判断「我是不是设备 0」 |
| ④ | `stablehlo.if`：**是** → 执行算子；**否** → 返回**零值** |
| ⑤ | token 类型用 `stablehlo.create_token` 生成 |

**核心思想**：单设备分片 = 「只有一台设备算，其他设备拿到占位值」。

### ② ★ `blessed operation`

**问题**：从「未归约」变成「已归约」**不能由普通算子顺手完成**。

**错误信息**：`'stablehlo.add' op dropped unreduced axis 'y' without a blessed operation
(e.g., sdy.reshard). This is an invalid transition from unreduced to reduced.`

**五条规则**：

| 规则 | 内容 |
|---|---|
| ① | 不能**悄悄丢掉**未归约轴 |
| ② | **blessed**（`sdy.reshard` / `sdy.sharding_constraint`）可以显式完成转换 |
| ③ | 函数调用**两端必须匹配**（4 个用例：实参/结果 丢掉/多出/缺少） |
| ④ | `func.return` 与 `manual_computation` **同样受约束**（规则统一） |
| ⑤ | 归约算子（`sum`/`max`/`min`）**不能变** —— 它们代表完全不同的部分结果 |

**验收点答案**：
- **从哪来**：`dot` / `reduce` 沿**归约维切分**时**引入**（L4-04 / L4-05）。
- **必须在哪消除**：由 **blessed operation** **显式**消除。

### ③ CMV1 移除

| 情形 | 行为 |
|---|---|
| **独立的** `all_gather` | **保留**（它是函数最终结果，没有使用者能「吸收」它） |
| **后面跟着 `dot` 的** | **移除**（唯一使用者是 `dot`，CMV1 能在 `dot` 内部自己聚合） |

**与 L4-10 的对比**：L4-10 是**通用**优化；本课是**后端特定**的适配。

### 三个 pass 的分工

| pass | 做什么 |
|---|---|
| `resolve_single_device_sharding` | 加守卫（**改写** IR） |
| `verify_unreduced_axes` | 报错（**不改** IR）—— 纯校验 |
| `remove_ag_rs_for_cmv1` | 移除（**改写** IR，后端特定） |

**一句话**：导出期的最后阶段要把所有「**特殊状态**」处理干净。

## 练习

见第 6 幕。三道题分别考单设备守卫、未归约的来去、CMV1 移除。

## 验收点

- [x] `check_ir_fidelity.py`：23 个 IR 块全部逐字来自声明的源文件
- [x] `check_coverage.py`：3 个源文件均被声明
- [x] `check_flags.py`：无非法 flag
- [x] `lint_lessons.py`：语法检查通过
- [x] `check_render.py`：无 JS 错误、无布局溢出、交互可用
- 自检：能说出 unreduced 轴从哪来、必须在哪被消除
