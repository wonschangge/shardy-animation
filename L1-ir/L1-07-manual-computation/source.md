<!-- sdy-coverage
ir/test/manual_computation_parse_print.mlir
ir/test/manual_computation_verification.mlir
ir/test/manual_computation_canonicalization.mlir
-->

# L1-07 · manual-computation — 源 IR

本课覆盖三个测试文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/manual_computation_parse_print.mlir` | 246 | 15 个合法用例：从空体到嵌套、动态形状、token |
| `ir/test/manual_computation_verification.mlir` | 296 | 21 条不变量校验错误 |
| `ir/test/manual_computation_canonicalization.mlir` | 89 | 7 个规范化用例 |

---

## 一、网格

```mlir
sdy.mesh @meshA = <["a"=2, "b"=2]>
```

```mlir
sdy.mesh @meshB = <["a"=4]>
```

```mlir
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
```

---

## 二、最简形态：什么都没有

```mlir
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    sdy.return
  } : () -> ()
```

带 manual 轴但体为空：

```mlir
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={"x", "y"} () {
    sdy.return
  } : () -> ()
```

---

## 三、核心用例：一个输入一个输出，都分片

外层张量是 `tensor<16x32xf32>`，`manual_axes={"a"}`（`"a"` 大小 2），
所以**区域内**的块参数是局部形状 `tensor<8x32xf32>`（16 ÷ 2 = 8）。

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

---

## 四、自由轴：manual 之外的轴仍可标注

`manual_axes={"a"}` 只冻结 `"a"`；`"b"` 是**自由轴**，可以在 in_sharding 里出现，
但必须排在 manual 轴**之后**（更 minor）。

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", "b"}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

注意输入 dim0 是 `{"a", "b"}` 而输出是 `{"a", ?}` —— 区域内的局部形状仍然只由 **manual 轴**决定，
`"b"` 是自由轴，不参与局部形状计算。

---

## 五、嵌套：各自绑定不同的 manual 轴

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@meshA, [{"b", ?}, {?}]>] out_shardings=[<@meshA, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg2: tensor<4x32xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x32xf32>
      sdy.return %2 : tensor<4x32xf32>
    } : (tensor<8x32xf32>) -> tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<8x32xf32>
```

外层绑定 `"a"`（16→8），内层绑定 `"b"`（8→4）。**嵌套的 manual 轴必须互不相同**。

---

## 六、动态形状与 token

动态维度同样按 manual 轴切分：

```mlir
    in_shardings=[<@meshA, [{"a", ?}, {?}]>]
    out_shardings=[<@meshA, [{"a", ?}, {?}]>]
    manual_axes={"a"} (%arg1: tensor<?x32xf32>) {
```

```mlir
    in_shardings=[<@meshA, []>, <@meshA, [{"b"}]>]
    out_shardings=[<@meshA, []>, <@meshA, [{"b"}]>]
    manual_axes={"b"} (%arg2: !stablehlo.token, %arg3: tensor<1xi64>) {
```

token 的分片必须是 rank 0（`[]`），这是 L1-02 的规则。

---

## 七、校验错误（21 条，节选）

### 1. 所有 in/out 分片必须绑定同一个网格

网格 `@meshA = <["a"=2]>` 与 `@meshB = <["a"=4]>`：

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{}, {}]>] out_shardings=[<@meshB, [{}, {}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
```

### 2. manual 轴必须来自该网格

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a"}, {}]>] out_shardings=[<@meshA, [{"a"}, {?}]>] manual_axes={"a", "b"} (%arg1: tensor<8x32xf32>) {
```

### 3. 分片项数必须等于值的个数

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
```

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
```

### 4. manual 轴不得引入 padding（维大小必须整除轴大小）

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<1xf32>) {
```

### 5. 局部形状必须与"外层形状 ÷ manual 轴大小"一致

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={} (%arg1: tensor<8x32xf32>) {
```

### 6. 操作数/结果个数必须与区域参数/返回值个数一致

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>, %arg2: tensor<16x32xf32>) {
```

### 7. manual 轴必须排在自由轴之前

```mlir
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"a", "b"}]>] out_shardings=[<@mesh, [{}, {}], replicated={"b"}>] manual_axes={"b"} (%arg1: tensor<16x16xf32>) {
```

### 8. 区域内不能引用区域外的值

```mlir
    %1 = stablehlo.add %arg0, %arg1 : tensor<16x32xf32>
```

### 9. 轴不能被父级 manual_computation 重复绑定

```mlir
    %2 = sdy.manual_computation(%arg2) in_shardings=[<@foo, [{}, {}]>] out_shardings=[<@foo, [{}, {}]>] manual_axes={"a"} (%arg3: tensor<4x32xf32>) {
```

---

## 八、规范化（7 个用例）

### 1. 删除未使用的参数

输入用三个参数、只返回其中一个：

```mlir
  %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"a"}]>, <@mesh, [{"a"}, {}]>, <@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg3: tensor<4xf32>, %arg4: tensor<16x32xf32>, %arg5: tensor<8xf32>) {
    sdy.return %arg4 : tensor<16x32xf32>
  } : (tensor<8xf32>, tensor<32x32xf32>, tensor<16xf32>) -> tensor<32x32xf32>
```

期望输出只剩一个参数：

```mlir
  // CHECK-NEXT: sdy.manual_computation(%arg1)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a"}, {}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a"}, {}]>]
  // CHECK-SAME:     manual_axes={"a"} (%arg3: tensor<16x32xf32>) {
```

### 2. 空体且无 manual 轴 → 直接内联

```mlir
  sdy.manual_computation() in_shardings=[] out_shardings=[]
      manual_axes={} () {
    stablehlo.custom_call @foo() {has_side_effect = true} : () -> ()
    sdy.return
  } : () -> ()
```

期望把区域内的算子提到外面：

```mlir
  // CHECK-NEXT: stablehlo.custom_call @foo() {has_side_effect = true} : () -> ()
  // CHECK-NEXT: return %arg0 : tensor<8xf32>
```
