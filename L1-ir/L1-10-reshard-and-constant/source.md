<!-- sdy-coverage
ir/test/reshard_verification.mlir
ir/test/reshard_canonicalization.mlir
ir/test/constant_parse_print.mlir
ir/test/constant_verification.mlir
-->

# L1-10 · reshard-and-constant — 源 IR

本课覆盖四个文件，共 351 行：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/reshard_verification.mlir` | 66 | 6 个用例（5 条报错 + 1 个合法反例） |
| `ir/test/reshard_canonicalization.mlir` | 255 | 17 个规范化用例：链式折叠 + CSE |
| `ir/test/constant_parse_print.mlir` | 15 | 2 个打印用例 |
| `ir/test/constant_verification.mlir` | 15 | 2 条校验错误 |

---

## 一、`sdy.reshard` 的基本形态与校验

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`

### 1. 分片本身必须合法（与普通张量分片同一套校验）

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"b"}], replicated={"a"}> : tensor<8xf32>
```

### 2. 区域内不能再用 manual 轴（两种写法都报）

```mlir
    %1 = sdy.reshard %arg1 <@mesh, [{"a"}, {}]> : tensor<8x32xf32>
```

```mlir
    %1 = sdy.reshard %arg1 <@mesh, [{}, {}], replicated={"a"}> : tensor<8x32xf32>
```

### 3. 不能改"保留的未归约轴"的归约算子

输入 `unreduced=max{"x"}`，reshard 后仍保留 `"x"` 但写成默认（sum）：

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x"}> : tensor<8xf32>
```

保留 `"x"` 的同时又引入 `"y"`，同样报错：

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x", "y"}> : tensor<8xf32>
```

### 4. 反例：丢掉 max 轴、引入新的未归约轴 —— 合法

输入 `unreduced={"x", "y"}`（默认 sum），输出 `unreduced={"x", "z"}`：
保留的 `"x"` 归约算子没变，丢掉了 `"y"`，新增了 `"z"` —— 合法。这个用例**没有** `expected-error`。

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{}], unreduced={"x", "z"}> : tensor<8xf32>
```

---

## 二、`reshard` 的规范化（17 个用例）

网格：`sdy.mesh @mesh = <["a"=2, "b"=2]>`、`sdy.mesh @mesh_4 = <["a"=4]>`

### 1. 链式折叠：中间结果无其它使用者 → 只留最后一个

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
```

期望折叠成一个 reshard：

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
  // CHECK-NEXT: return %0
```

### 2. 但中间结果还有别的使用者时，不能折叠

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]>
  // CHECK-NEXT: return %0, %1
```

### 3. 三段链式同样折叠

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %2 = sdy.reshard %1 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]>
  // CHECK-NEXT: return %0
```

### 4. CSE：同一个输入 + 同一个目标分片 → 合并成一个

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %1 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
  // CHECK-NEXT: return %0, %0
```

### 5. 目标分片不同 → 不合并

```mlir
  %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %1 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
```

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
  // CHECK-NEXT: %1 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]>
  // CHECK-NEXT: return %0, %1
```

### 6. 合并后多个使用者共享同一个 reshard

```mlir
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
  // CHECK-NEXT: %1 = stablehlo.sine %0 : tensor<8x8xf32>
  // CHECK-NEXT: %2 = stablehlo.cosine %0 : tensor<8x8xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %0 : tensor<8x8xf32>
  // CHECK-NEXT: return %1, %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
```

---

## 三、`sdy.constant`

### 1. 基本打印

```mlir
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
```

### 2. 对比：量化类型由 `stablehlo.constant` 承载

```mlir
  %0 = stablehlo.constant() {value = dense<[1, 512, 4]> : tensor<3xi32>} : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>
```

期望打印为：

```mlir
  // CHECK-NEXT: stablehlo.constant() <{value = dense<[1, 512, 4]> : tensor<3xi32>}> : () -> tensor<3x!quant.uniform<i32:f32, 2.000000e+00:15>>
```

### 3. 校验：必须有静态形状

动态形状：

```mlir
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x?xf32>
```

无 rank：

```mlir
  %0 = sdy.constant dense<1.000000e+00> : tensor<*xf32>
```

两者都报 `elements literal type must have static shape`。
