<!-- sdy-coverage
ir/test/collective_parse_print.mlir
ir/test/collective_verification.mlir
ir/test/collective_canonicalization.mlir
-->

# L1-06 · collectives — 源 IR

本课覆盖三个文件，是 L1 最大的一课：

| 文件 | 行数 | 作用 |
|---|---|---|
| `ir/test/collective_parse_print.mlir` | 508 | 8 个集合通信算子、71 个实例：语法、子轴语义、跨网格 |
| `ir/test/collective_verification.mlir` | 1160 | 112 条校验错误 |
| `ir/test/collective_canonicalization.mlir` | 399 | 42 个规范化用例：消除空通信、融合 |

**八个算子**：`sdy.all_gather`、`sdy.all_slice`、`sdy.all_to_all`、`sdy.all_reduce`、
`sdy.reduce_scatter`、`sdy.collective_permute`、`sdy.sharded_to_unreduced`、`sdy.replicated_to_unreduced`。

---

## 一、网格

```mlir
sdy.mesh @mesh1 = <["x"=2, "y"=2]>
```

```mlir
sdy.mesh @mesh1_non_iota = <["x"=2, "y"=2], device_ids=[3, 2, 1, 0]>
```

```mlir
sdy.mesh @mesh2 = <["x"=2, "y"=2, "z"=2]>
```

```mlir
sdy.mesh @mesh3 = <["x"=4, "y"=2]>
```

```mlir
sdy.mesh @mesh4 = <["x"=8, "y"=2, "z"=2]>
```

```mlir
sdy.mesh @mesh6 = <["x"=4, "y"=4]>
```

```mlir
sdy.mesh @mesh7 = <["x"=16, "y"=2]>
```

---

## 二、`all_gather` / `all_slice`：一对逆操作

`all_gather` 把别的设备上的分片收过来（分片 → 复制）；`all_slice` 从整份里取自己那片（复制 → 分片）。
参数是**逐维的轴列表**：外层列表项数 = 张量 rank。

### 1. 一维 gather / 一维 slice

输入 `[{"y"}, {"x"}]`，gather 第 1 维的 `"x"`：

```mlir
%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {}]> : tensor<16x8xf32>
```

逆操作：

```mlir
%0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> : tensor<16x8xf32>
```

### 2. 一次 gather 多个维度

```mlir
%0 = sdy.all_gather [{"x"}, {"z"}] %arg0 out_sharding=<@mesh2, [{"y"}, {}]> : tensor<16x8xf32>
```

### 3. 同一维上 gather 多个轴

```mlir
%0 = sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]> : tensor<16x8xf32>
```

---

## 三、`all_to_all`：把分片从一个维度搬到另一个维度

参数语法是 `{轴}: 源维度 -> 目标维度`，多个参数用逗号分隔，源维度必须升序。

### 1. 单个参数

```mlir
%0 = sdy.all_to_all [{"x"}: 0->1] %arg0 out_sharding=<@mesh1, [{}, {"x"}]> : tensor<16x8xf32>
```

### 2. 一次搬多个轴

```mlir
%0 = sdy.all_to_all [{"y", "x"}: 0->1] %arg0 out_sharding=<@mesh2, [{"z"}, {"y", "x"}]> : tensor<16x8xf32>
```

### 3. 子轴也可以搬

```mlir
%0 = sdy.all_to_all [{"x":(4)2, "z"}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]> : tensor<16x8xf32>
```

---

## 四、`all_reduce` / `reduce_scatter`：部分和的处理

### 1. 基本归约（默认 sum）

```mlir
%0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
```

### 2. 指定归约算子

```mlir
%0 = sdy.all_reduce max {"y"} %arg0 out_sharding=<@mesh1, [{}, {"x"}]> :  tensor<16x2xf32>
```

### 3. 归约掉一个未归约轴：该轴变为**显式复制**

```mlir
%0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh2, [{}, {"x"}], replicated={"y"}, unreduced={"z"}> :  tensor<16x2xf32>
```

### 4. `reduce_scatter` = 归约 + 切片

```mlir
%0 = sdy.reduce_scatter [{"y"}] %arg0 out_sharding=<@mesh1, [{"x", "y"}]> : tensor<16xf32>
```

### 5. 多维形式

```mlir
%0 = sdy.reduce_scatter [{}, {"y"}] %arg0 out_sharding=<@mesh1, [{"x"}, {"y"}]> : tensor<16x8xf32>
```

### 6. 空的轴列表 = 不做任何事（可被规范化消除）

```mlir
%0 = sdy.reduce_scatter [{}, {}] %arg0 out_sharding=<@mesh1, [{"x"}, {}]> : tensor<16x8xf32>
```

---

## 五、`collective_permute`：重排或替换轴

约束：**每一维的分片大小必须保持不变**，只是换了哪些轴来切。

### 1. 两个维度之间互换轴

```mlir
%0 = sdy.collective_permute %arg0 out_sharding=<@mesh2, [{"y"}, {"x"}]> : tensor<16x8xf32>
```

### 2. 换到另一个网格（设备顺序不同）

```mlir
%0 = sdy.collective_permute %arg0 out_sharding=<@mesh1_non_iota, [{"x", "y"}, {}]> : tensor<16x8xf32>
```

---

## 六、未归约状态的两个操作

`sdy.sharded_to_unreduced`：把**已分片**的轴变成未归约（等价于"先 all-gather 再多份求和"的中间态）。

```mlir
%0 = sdy.sharded_to_unreduced [{}, {"x"}] %arg0 out_sharding=<@mesh1, [{"y"}, {}], unreduced={"x"}> : tensor<16x8xf32>
```

`sdy.replicated_to_unreduced`：把**复制**的轴变成未归约。

```mlir
%0 = sdy.replicated_to_unreduced {"x", "y"} %arg0 out_sharding=<@mesh1, [{}, {}], unreduced=min{"x", "y"}> : tensor<16x8xf32>
```

---

## 七、子轴在集合通信中的四种语义

| 写法 | 语义 |
|---|---|
| `exact_match` | 通信用到的子轴与分片里的**完全一致** |
| `ignored` | 通信的轴与分片里的轴**不相干**，各自保留 |
| `suffix_of_full` | 子轴是**完整轴的后缀**，通信只动其中一部分 |
| `suffix_of_subaxis` | 子轴是**另一个子轴的后缀** |

### exact_match

```mlir
%0 = sdy.all_gather [{}, {"x":(1)2}] %arg0 out_sharding=<@mesh3, [{"y"}, {}]> : tensor<16x8xf32>
```

### ignored

```mlir
%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh3, [{}, {"x":(1)2}]> : tensor<16x8xf32>
```

### suffix_of_full

```mlir
%0 = sdy.all_gather [{}, {"x":(4)2, "z"}] %arg0 out_sharding=<@mesh4, [{"y"}, {"x":(1)4}]> : tensor<16x8xf32>
```

### suffix_of_subaxis（all_to_all 版本）

```mlir
%0 = sdy.all_to_all [{"x":(4)2, "z"}: 1->0] %arg0 out_sharding=<@mesh4, [{"y", "x":(4)2, "z"}, {"x":(1)4}]> : tensor<16x8xf32>
```

---

## 八、校验错误（112 条，节选）

网格：`sdy.mesh @mesh = <["x"=2, "y"=2]>`

### 1. 操作数必须有分片

```mlir
%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

### 2. 结果网格必须与操作数一致

```mlir
%0 = sdy.all_gather [{}, {"b"}] %arg0 out_sharding=<@mesh1, [{"y"}, {"x"}]> :  tensor<16x8xf32>
```

### 3. 轴引用不能重复 / 相邻子轴必须合并

```mlir
%0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh, [{}, {"x", "x"}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_gather [{}, {"x":(1)2, "x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

### 4. 集合通信轴的 rank 必须等于张量 rank

```mlir
%0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
```

### 5. 轴必须真的能"应用"到操作数分片上

```mlir
%0 = sdy.all_gather [{}, {"x", "y"}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_gather [{"x", "y"}, {}] %arg0 out_sharding=<@mesh, [{}, {}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_gather [{}, {"x":(2)2}] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

### 6. 结果分片必须与"推导出的分片"一致

```mlir
%0 = sdy.all_gather [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
```

### 7. all_to_all 的参数约束

```mlir
%0 = sdy.all_to_all [] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_to_all [{"y"}: 2->1] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_to_all [{"y"}: 0->-1] %arg0 out_sharding=<@mesh, [{}, {"y"}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_to_all [{"y"}: 0->0] %arg0 out_sharding=<@mesh, [{"y"}, {}]> :  tensor<16x8xf32>
```

```mlir
%0 = sdy.all_to_all [{"y"}: 2->1, {"x"}: 0->3] %arg0 out_sharding=<@mesh, [{}, {"y"}, {}, {"x"}]> :  tensor<16x8x8x8xf32>
```

---

## 九、规范化：消除与融合

### 1. 全零的 `collective_permute` 没有使用者 → 直接删除

（`null_collective_permute` 用例验证此类消除。）

### 2. `all_reduce` + `all_slice` → `reduce_scatter`

输入是两个连续的算子：

```mlir
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<16x2xf32>
  %1 = sdy.all_slice [{}, {"x"}] %0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
```

期望融合成一个：

```mlir
  // CHECK-NEXT: %0 = sdy.reduce_scatter [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
```

### 3. 链式 `all_to_all` 可以合并

（`all_to_all_fusion_chained_ops`、`all_to_all_fusion_two_ops_need_sorting` 等用例覆盖。）
