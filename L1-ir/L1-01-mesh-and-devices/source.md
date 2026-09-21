<!-- sdy-coverage
ir/test/mesh_parse_print.mlir
ir/test/mesh_verification.mlir
-->

# L1-01 · mesh-and-devices — 源 IR

本课覆盖两个测试文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/mesh_parse_print.mlir` | 31 | 10 个合法 `sdy.mesh`，验证解析与**打印规范化** |
| `ir/test/mesh_verification.mlir` | 34 | 8 个非法 `sdy.mesh`，验证校验器报错 |

下面每条都是**逐字**摘自上述文件。测试里的 `// CHECK:` 行表示期望的打印结果，
`// expected-error` 行表示期望的报错。

---

## 一、空网格与 maximal-sharding 网格

空网格 `<[]>` 是一个**占位符**：它没有轴，传播时会被替换成真实网格。

```mlir
sdy.mesh @empty_mesh = <[]>
```

如果只给一个设备 ID 而不给轴，得到的是 **maximal-sharding 网格** —— 表示"这台设备独占全部"：

```mlir
sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]>
```

```mlir
sdy.mesh @maximal_mesh_3 = <[], device_ids=[3]>
```

---

## 二、带轴的网格（隐式设备编号）

只写轴、不写 `device_ids` 时，设备编号默认是 `iota(product(轴大小))`，即行优先编号。

```mlir
sdy.mesh @single_axis_of_size_1 = <["a"=1]>
```

```mlir
sdy.mesh @single_axis_of_size_2 = <["a"=2]>
```

```mlir
sdy.mesh @two_axes = <["a"=2, "b"=1]>
```

---

## 三、带轴的网格（显式设备顺序）

`device_ids` 允许把逻辑位置映射到任意设备顺序。

```mlir
sdy.mesh @single_axis_explicit_device_ids = <["a"=2], device_ids=[1, 0]>
```

```mlir
sdy.mesh @two_axes_explicit_device_ids = <["a"=2, "b"=1], device_ids=[1, 0]>
```

---

## 四、打印规范化：iota 会被省略

输入显式写了 `device_ids=[0, 1, 2, 3]`，但因为**它正好等于默认的 iota 顺序**，
打印时会被省略。测试的 CHECK 行体现了这一点：

```mlir
sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2], device_ids=[0, 1, 2, 3]>
```

期望打印为（CHECK 行）：

```mlir
sdy.mesh @iota_explicit_device_ids = <["a"=2, "b"=2]>
```

同理，轴大小全为 1 时 `device_ids=[0]` 也是 iota，同样被省略：

```mlir
sdy.mesh @single_axis_of_size_1_with_device_id = <["a"=1], device_ids=[0]>
```

---

## 五、校验错误（8 种）

### 1. 轴大小必须 ≥ 1

```mlir
sdy.mesh @mesh = <["a"=2, "b"=0]>
```

### 2. 轴名不可重复

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2, "a"=4]>
```

### 3. 设备 ID 不可为负

```mlir
sdy.mesh @mesh = <[], device_ids=[-1]>
```

### 4. 无轴时 device_ids 至多一个元素

```mlir
sdy.mesh @mesh = <[], device_ids=[1, 0]>
```

### 5. 轴大小之积必须等于 device_ids 个数（多给）

```mlir
sdy.mesh @mesh = <["a"=2, "b"=2], device_ids=[0, 2, 1, 3, 4, 5]>
```

### 6. 轴大小之积必须等于 device_ids 个数（少给）

```mlir
sdy.mesh @mesh = <["a"=2], device_ids=[0]>
```

### 7. 设备 ID 不可重复

```mlir
sdy.mesh @mesh_duplicated_device_ids = <["a"=2], device_ids=[1, 1]>
```

### 8. 排序后的设备 ID 必须是 iota（越界）

```mlir
sdy.mesh @mesh_out_if_bound_device_ids = <["a"=2], device_ids=[2, 1]>
```

---

## 附：测试的 RUN 行

```
// RUN: sdy_opt %s 2>&1 | FileCheck %s
// RUN: sdy_opt %s -split-input-file -verify-diagnostics
```
