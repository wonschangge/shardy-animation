<!-- sdy-coverage
ir/test/sharding_constraint_verification.mlir
ir/test/sharding_group_parse_print.mlir
ir/test/propagation_barrier_parse_print.mlir
ir/test/propagation_barrier_verification.mlir
-->

# L1-09 · constraint-group-barrier — 源 IR

本课覆盖四个文件，共 107 行（L1 最小一课）：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/sharding_constraint_verification.mlir` | 60 | 5 条校验错误（含 2 条 manual 绑定冲突） |
| `ir/test/sharding_group_parse_print.mlir` | 8 | 1 个用例：`group_id` |
| `ir/test/propagation_barrier_parse_print.mlir` | 32 | 4 个方向用例 |
| `ir/test/propagation_barrier_verification.mlir` | 7 | `BOTH` 被拒绝 |

---

## 一、`sdy.sharding_constraint`

网格：`sdy.mesh @mesh = <["a"=2,"b"=2]>`

### 1. 分片本身必须合法（与普通张量分片同一套校验）

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>
```

### 2. 区域内不能再用 manual 轴

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a",?}, {?}]>] out_shardings=[<@mesh, [{"a",?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>
```

### 3. 把它放进 replicated 也不行

```mlir
    %1 = sdy.sharding_constraint %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32>
```

### 4. token 只能 rank 0

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : !stablehlo.token
```

### 5. 不能改已有未归约轴的归约算子

输入已经是 `unreduced=max{"x"}`，约束却写成默认（sum）：

```mlir
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>
```

---

## 二、`sdy.sharding_group`

```mlir
  sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>
```

### ⚠ 上游测试文件里的一个陷阱

该文件的 CHECK 行是：

```mlir
  // CHECK sdy.sharding_group %arg0 group_id=21 type=AS  : tensor<8xf32>
```

这一行**缺少冒号**（`// CHECK` 而非 `// CHECK:`），所以 FileCheck **不把它当指令**，
它只是一句普通注释。它声称输出里会有 `type=AS`，但实测 `sdy_opt` 的输出是：

```mlir
  sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>
```

并没有 `type=AS` —— 该属性在当前 `Sdy_ShardingGroupOp` 定义中不存在
（`let arguments = (ins AnyRankedTensor:$input, I64Attr:$group_id);`）。
这条注释是**过时残留**，不要据此推断语法。

（本课件的验证方式：原样跑 `sdy_opt | FileCheck` 退出码为 0；把那行补上冒号变成真指令后
退出码为 1 且报 `expected string not found`。两相对照可确认它不是生效的指令。）

---

## 三、`sdy.propagation_barrier`

网格：`sdy.mesh @mesh = <["a"=2,"b"=2]>`

### 1. 三个合法方向

```mlir
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8xf32>
```

```mlir
  %0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8xf32>
```

```mlir
  %0 = sdy.propagation_barrier %arg0 allowed_direction=NONE : tensor<8xf32>
```

### 2. 屏障自身也可以带分片

```mlir
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a",?}]>]>} : tensor<8xf32>
```

### 3. `BOTH` 被拒绝

```mlir
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BOTH : tensor<8xf32>
```

（报文：`cannot specify \`BOTH\` as the direction` —— 允许双向就等于没有屏障。）
