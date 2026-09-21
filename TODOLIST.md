# animation/TODOLIST.md — SDY 测试 IR 全覆盖动画化计划

> **目标**：把 `shardy/dialect/sdy` 下**全部测试 IR** 做成由浅入深、可逐课验收的动画讲解，
> 让学习者从"看懂语法"一路走到"能读懂分区器产出的设备代码"。
>
> **范围**：`shardy/dialect/sdy/**/test/**/*.mlir` — **241 个文件 / 45,177 行**
> （不含 `*_test.cc` 单测，理由见 §7.1）
>
> **硬性验收**：§3 矩阵中每一行的文件，都必须能在对应课件的 `source.md` 中找到**逐字引用**。
> 本文所有文件名均**逐个枚举**，不使用 `*` 通配，以便 `tools/check_coverage.py` 直接断言。

---

## 0. 现状

| 项 | 状态 |
|---|---|
| `animation/intro/` | ✅ 已完成：20 幕入门动画（分片表示 / 传播 / 集合通信 / SPMD / MPMD 概览） |
| 动画引擎 | ✅ 已存在，但内嵌在 `intro/app.js`，需提炼为共享模块 |
| 本计划 | 📋 待执行 |

`intro/` 定位为**预备篇**：看完知道 Shardy 是什么。本计划是**精读篇**：逐课啃测试 IR。

---

## 1. 总体设计

### 1.1 目录结构

```
animation/
├── TODOLIST.md                 # 本文件
├── README.md                   # 【新建】总索引：学习路线图 + 课件目录 + 术语表
├── shared/                     # 【新建】共享动画引擎（从 intro 提炼，单一来源）
│   ├── engine.js               #   舞台缩放 / 时间轴 / 场景挂载 / 键鼠控制
│   ├── widgets.js              #   设备网格 / 张量网格 / 轴 chip / IR 高亮 / 因子表
│   ├── theme.css               #   主题、版式、动画
│   └── lesson-shell.js         #   课件外壳：自动生成侧栏 / 进度条 / 目录跳转
├── tools/                      # 【新建】校验脚本（见 §5）
│   ├── check_render.py         #   逐课件渲染 + console 错误 + 布局溢出 + 多分辨率
│   ├── check_coverage.py       #   241 个测试文件 → 课件 的覆盖度断言
│   ├── check_ir_fidelity.py    #   课件中 IR 片段与源测试文件逐字比对
│   └── check_flags.py          #   课件中 pass flag 与 passes.td 比对
│
├── intro/                      # ✅ 预备篇（已完成，20 幕）
│
├── L1-ir/                      # 第 1 层：IR 构件（看语法）       10 课
├── L2-propagation/             # 第 2 层：传播算法（看算法）       12 课
├── L3-import/                  # 第 3 层：导入流水线（看准备）     11 课
├── L4-export-core/             # 第 4 层：导出核心（看补通信）     17 课
├── L5-lowering/                # 第 5 层：全局→局部（看降级）      9 课
├── L6-executable/              # 第 6 层：可执行验证（看数值正确性）11 课
└── L7-capstone/                # 第 7 层：综合实战                 4 课
```

**合计 74 课**，覆盖 241 个测试文件。

### 1.2 课件模板（每个课时一个目录）

```
L?-??-<slug>/
├── index.html      # 引用 ../../shared/*
├── lesson.js       # 本课场景定义（沿用 intro 的 SCENES 数组格式）
├── source.md       # 【验收依据】逐字引用的测试文件清单 + IR 片段 + 文件路径
└── README.md       # 学习目标 / 前置课 / 练习 / 验收点
```

`index.html` 必须能**双击直接打开**（纯静态、无 CDN、离线可用）—— 与 `intro` 一致。

### 1.3 每课的固定三段结构

1. **看 IR**（Read）：展示源测试文件的真实输入 IR，逐行标注。
2. **看动画**（Animate）：把该 pass 的变换可视化（before → 中间态 → after）。
3. **做练习**（Practice）：给一个小 IR，让学习者预测输出，点击揭晓。

### 1.4 难度分层原则

| 层 | 认知目标 | 看完能… |
|---|---|---|
| L1 | **认识构件** | 读懂任何 SDY IR，知道每个 op/attr 是什么 |
| L2 | **理解算法** | 预测传播结果，看懂 `debug-sharding-origins` 输出 |
| L3 | **理解准备** | 知道传播前程序被做了哪些规范化，为什么 |
| L4 | **理解代价** | 知道通信从哪来、reshard 怎么变成 collective |
| L5 | **理解降级** | 看懂 `tensor<8x8>` 怎么变成 `tensor<4x4>` |
| L6 | **验证正确** | 能自己加一个可执行测试，验证分区结果数值正确 |
| L7 | **综合运用** | 独立分析真实模型的并行策略 |

---

## 2. 分层细项计划

> 优先级：**P0** 主干必做 / **P1** 应做 / **P2** 加分。
> 每课的"验收"默认含 §5 的 A/B/C 组自动检查，此处只列**本课特有**的验收点。
> 文件名省略 `.mlir` 后缀；`source.md` 中需写全路径。

---

### L1 — IR 构件（第 1 层 · 看语法）

**层目标**：把 SDY 方言每个构件讲透，看完能读懂仓库里任何 `.mlir`。
**覆盖**：`ir/test/` — 28 文件 / 5,048 行
**前置**：`intro/` 第 3～9 幕

- [x] **L1-01 `mesh-and-devices`** — P0
  - 覆盖（2）：`mesh_parse_print` `mesh_verification`
  - 讲解：命名轴；`device_ids` 自定义设备顺序；空网格 `<[]>`；maximal-sharding 网格 `<[], device_ids=[3]>`；
    为什么 `product(axis_sizes)` 必须等于设备数；报错 `duplicate axis name: "a"`、`total product of axis sizes must match total number of device ids, got: 4 != 6`
  - 验收：能解释 `<["a"=3,"b"=2], device_ids=[0,2,4,1,3,5]>` 与 `<["a"=3,"b"=2]>` 的语义差别

- [x] **L1-02 `tensor-sharding-syntax`** — P0
  - 覆盖（2）：`tensor_sharding_parse_print` `tensor_sharding_parsing_failure`
  - 讲解：`#sdy.sharding` 完整语法 → 维分片 / 开闭维 `?` / `replicated=` / `unreduced=` / `reduction_op`；
    子轴 `"x":(2)4` 的 pre-size 语义；优先级 `[{"a"}p0, {"b"}p1]`；解析失败的信息如何定位
  - 验收：给定 `[{"a"}, {"b":(2)2}] replicated={"c"}`，能算出局部形状并判断是否合法

- [x] **L1-03 `tensor-sharding-verification`** — P0
  - 覆盖（1）：`tensor_sharding_verification`（556 行，L1 最大）
  - 讲解：按错误类别组织动画：重复轴 `duplicate axis ref: "a"`、
    相邻子轴可合并 `two consecutive sub-axes can be merged: "a":(2)2, "a":(4)4`、
    分片数与 rank 不符 `op result shardings don't match number of values: 3 shardings vs 2 values`、
    维度 0 不可分片、replicated/unreduced 的排序要求
  - 验收：给 5 段非法 sharding，各自说出违反哪条不变量

- [x] **L1-04 `edge-sharding`** — P1
  - 覆盖（2）：`edge_sharding_parse_print` `edge_sharding_verification`
  - 讲解：边分片用途（记录分片由哪个 operand/result 引入），与 `debug-edge-source-sharding` 的关系；为 L2-12 铺垫
  - 验收：能说明边分片与张量分片的区别及使用场景

- [x] **L1-05 `op-sharding-rule`** — P0
  - 覆盖（3）：`sharding_rule_parse_print` `sharding_rule_parsing_failure` `sharding_rule_verification`
  - 讲解：`#sdy.op_sharding_rule<([i,k],[k,j])->([i,j]) {i=8,j=16,k=8}>` 逐字段拆解；
    因子四分类（pass-through / reduction / need_replication / permutation）；
    `blocked_propagation_factors`；`is_custom_rule`；复合因子 `((ij), k)`
  - 验收：能为 `stablehlo.transpose` 手写一条正确的 sharding rule

- [x] **L1-06 `collectives`** — P0
  - 覆盖（3）：`collective_parse_print` `collective_verification` `collective_canonicalization`
  - 讲解：6 个集合通信算子的精确语法与约束；**`out_sharding` 是推导结果而非输入**；
    规范化（all-gather 的合并与消除）；为 L4-08 铺垫
  - 验收：给定输入分片与 `gathering_axes`，能推出 `out_sharding` 并判断是否合法

- [x] **L1-07 `manual-computation`** — P1
  - 覆盖（3）：`manual_computation_parse_print` `manual_computation_verification` `manual_computation_canonicalization`
  - 讲解：`manual_axes` 语义；区域内是**局部形状**；自由轴仍可传播；
    manual 轴必须在所有 in/out sharding 中显式出现（分片或显式复制）；manual 轴不得引入 padding；可嵌套
  - 验收：给定全局形状 + in_shardings + manual_axes，能算出 body 内 block argument 的类型

- [x] **L1-08 `named-computation-and-dataflow`** — P1
  - 覆盖（4）：`named_computation_parse_print` `named_computation_verification`
    `data_flow_edge_verification` `func_data_flow_edge_verification`
  - 讲解：为什么需要 `sdy.named_computation`（传播要能穿过函数调用）；
    数据流边 = source 集合 + target 集合 + owner；`ShardableDataFlowOpInterface`；
    while 的 n 条边如何划分（`x_i`/`return_value_i` → `y_i`/`pred_arg_i`/`body_arg_i`）
  - 验收：能画出 while / case / call 各自的数据流边

- [x] **L1-09 `constraint-group-barrier`** — P0
  - 覆盖（4）：`sharding_constraint_verification` `sharding_group_parse_print`
    `propagation_barrier_parse_print` `propagation_barrier_verification`
  - 讲解：悬空 vs 有使用者的 `sharding_constraint`；`sdy.sharding_group ... group_id=N`；
    `allowed_direction` 三态（FORWARD/BACKWARD/NONE）；三者生命周期（谁读谁写）
  - 验收：能判断某段 IR 中 constraint 是否会"直接约束张量本身"

- [x] **L1-10 `reshard-and-constant`** — P1
  - 覆盖（4）：`reshard_verification` `reshard_canonicalization` `constant_parse_print` `constant_verification`
  - 讲解：`sdy.reshard` 的生命周期（传播后出现 → 分区器消灭）；
    为什么 `sdy.constant` 故意不实现 ConstantLike、不带 folder
    （防止贪婪重写器把常量合并回去，破坏 per-use 分片）
  - 验收：能解释"为什么 Shardy 要自己定义一个 constant op"

---

### L2 — 传播算法（第 2 层 · 看算法）

**层目标**：理解传播如何从少量约束推出全图分片，并能读懂调试输出。
**覆盖**：`transforms/propagation/test/`(16) + `transforms/propagation/debugging/test/`(2) — 18 文件 / 8,844 行
**前置**：L1 全部

- [x] **L2-01 `basic-propagation`** — P0 ★ 最大单课
  - 覆盖（1）：`basic_propagation`（1106 行）
  - 讲解：按算子族拆成多条动画线（逐元素 / dot / reduce / reshape / broadcast / slice / gather / while…），
    每条给 before→after。重点：**开放维度的轴如何向后传播**、factor 空间的收集与扩展
  - 验收：随机抽 3 个用例，手工推出 pass 输出分片

- [x] **L2-02 `conservative-mode`** — P1
  - 覆盖（1）：`basic_propagation_conservative`
  - 讲解：`conservative-propagation` 禁止分裂轴与不可整除轴；同一 IR 开关对比动画
  - 验收：能说出保守模式在什么场景下是必须的

- [x] **L2-03 `keep-sharding-rules`** — P2
  - 覆盖（1）：`basic_propagation_keep_sharding_rules`
  - 讲解：`keep-sharding-rules` 保留推导出的 rule 用于调试
  - 验收：能读懂保留后的 IR 中的 rule 属性

- [x] **L2-04 `aggressive-propagation`** — P0
  - 覆盖（1）：`aggressive_propagation`
  - 讲解：与 basic 的差异 —— 冲突如何被"强行"消解、何时引入额外通信、`propagation-strategy` 选项
  - 验收：同一输入分别用 basic/aggressive 跑，说出每处差异的原因

- [x] **L2-05 `op-priority-propagation`** — P1
  - 覆盖（1）：`op_priority_propagation`
  - 讲解：算子按类型分批；方向选择规则 `BOTH > BACKWARD == FORWARD > NONE`；为何逐元素优先于 dot
  - 验收：能预测某算子在某一轮的传播方向

- [x] **L2-06 `user-priority-propagation`** — P0
  - 覆盖（1）：`user_priority_propagation`
  - 讲解：`p0/p1/p2` 分轮传播；低优先级开维不被高优先级覆盖；与 `intro` 第 10 幕呼应但深入到 IR 层
  - 验收：给定带 p0/p1/p2 标注的 IR，能画出每轮结束时的分片状态

- [x] **L2-07 `sharding-group-propagation`** — P1
  - 覆盖（1）：`sharding_group_propagation`
  - 讲解：组内"一荣俱荣"；无数据依赖时靠组把分片传过去
  - 验收：能解释没有 sharding_group 时该张量为何退化为全复制

- [x] **L2-08 `data-flow-edges`** — P1
  - 覆盖（2）：`basic_propagation_data_flow_edges` `basic_propagation_token`
  - 讲解：数据流边传播（视作恒等规则）；token 类型的特殊处理
  - 验收：能画出 while 每条数据流边并预测传播结果

- [x] **L2-09 `manual-computation-propagation`** — P1
  - 覆盖（1）：`basic_propagation_manual_computation`
  - 讲解：传播如何**穿过** manual_computation 在自由轴上继续工作；manual 轴为何被冻结
  - 验收：给定含 manual_computation 的 IR，指出哪些轴会被继续分片

- [x] **L2-10 `op-sharding-rule-registry`** — P0 ★ 工具课
  - 覆盖（3）：`op_sharding_rule_registry` `op_sharding_rule_registry_conservative` `op_sharding_rule_registry_failures`
  - 讲解：`-sdy-populate-op-sharding-rules` 一次性打印**所有**算子（stablehlo + sdy）的规则；
    做成可交互"规则速查表"：按算子族索引，点击看因子表
  - 验收：表中每个算子的规则能与 `-sdy-populate-op-sharding-rules` 实际输出对上

- [x] **L2-11 `propagation-pipeline`** — P0
  - 覆盖（3）：`propagation_pipeline` `propagation_pipeline_data_flow_edges` `propagation_pipeline_dedup_functions_fully_true`
  - 讲解：完整传播流水线内部顺序（user priority → op priority → aggressive → basic）；函数去重选项影响
  - 验收：能说明为何单独跑 `-sdy-basic-propagate` 与跑完整 pipeline 结果不同

- [x] **L2-12 `propagation-debugging`** — P0 ★ 实用课
  - 覆盖（2）：`debugging/test/edge_shardings` `debugging/test/sharding_origins`
  - 讲解：`debug-sharding-origins` 与 `debug-edge-source-sharding` 输出怎么读；
    做成"分片来源追溯"动画：每个张量的分片可点开看它从哪来
  - 验收：给定一个张量的分片，能用调试属性说出它被哪个输入"传染"

---

### L3 — 导入流水线（第 3 层 · 看准备）

**层目标**：理解传播前程序被做了哪些规范化，以及"假依赖"为什么必须消除。
**覆盖**：`transforms/import/test/`(18) + `transforms/import/test/executable_explicit_gather_scatter_batching/`(5) — 23 文件 / 6,712 行
**前置**：L1、L2-01

- [x] **L3-01 `import-pipeline-overview`** — P0
  - 覆盖（1）：`import_pipeline`
  - 讲解：导入流水线完整 pass 顺序与每步目的；总览动画
  - 验收：能背出导入流水线的阶段划分

- [x] **L3-02 `constant-splitter`** — P0 ★ 本层最大课
  - 覆盖（1）：`constant_or_scalar_splitter`（1782 行，全计划最大文件）
  - 讲解："假依赖"问题：两个不相干的使用者共用常量，不该被强行切成同一种分片；
    常量子计算的递归定义；标量扩展；为什么标量张量不参与拆分
  - 验收：给定共用常量的程序，指出哪些常量子计算会被拆成几份

- [x] **L3-03 `apply-sharding-constraints`** — P0
  - 覆盖（2）：`apply_sharding_constraints` `apply_sharding_constraints_preinlined`
  - 讲解：把 constraint 分片**真正写进输入张量**的三个条件；数据流边目标时的特殊处理；constraint 链的替换规则
  - 验收：能判断某条 constraint 会被"落实"还是"留给传播"

- [x] **L3-04 `add-data-flow-edges`** — P1
  - 覆盖（1）：`add_data_flow_edges`
  - 讲解：为每个数据流边拥有者插入 `sdy.data_flow_edge`；已有分片如何被继承
  - 验收：能预测插入后 IR 的结构

- [x] **L3-05 `add-func-data-flow-edges`** — P1
  - 覆盖（1）：`add_func_data_flow_edges`
  - 讲解：函数参数与调用结果的边；为跨函数传播铺路
  - 验收：能说出 func 边与普通边的区别

- [x] **L3-06 `import-func-calls`** — P2
  - 覆盖（3）：`import_func_calls` `import_func_calls_add_data_flow_edges_on_named_computations_false` `flatten_call_graph`
  - 讲解：call → named_computation 内联；多调用点时函数体克隆；调用图展平
  - 验收：能解释同名函数被多次调用时的处理策略

- [x] **L3-07 `lift-inlined-meshes`** — P1
  - 覆盖（1）：`lift_inlined_meshes`
  - 讲解：内联 `MeshAttr` 提升为 `sdy.mesh` 符号；去重与命名规则（`maximal_mesh_{id}` / `mesh` / `mesh_N`）
  - 验收：能预测新 mesh 的符号名

- [x] **L3-08 `manual-axes-cleanup`** — P2
  - 覆盖（2）：`manual_axes_cleanup` `manual_axes_cleanup_failures`
  - 讲解：补齐 in/out sharding 中缺失的 manual 轴到 replicated；按网格序排序；空 region 报错
  - 验收：能指出哪条 IR 会触发报错以及为什么

- [x] **L3-09 `sharding-group-import`** — P1
  - 覆盖（2）：`sharding_group_import` `sharding_group_constraints`
  - 讲解：组传递闭包合并（一个张量同属 G1、G2 ⇒ 合并）；组 id 规范化为 0..N-1；嵌套块约束
  - 验收：给定多个重叠分组，算出合并后的分组

- [x] **L3-10 `misc-import-cleanup`** — P2
  - 覆盖（3）：`remove_size_one_axes` `pre_order_funcs` `propagate_sharding_from_func_to_call`
  - 讲解：去掉 size=1 轴以免冲突；函数按调用图前序重排（模拟自顶向下传播）；函数结果分片回填到调用点
  - 验收：能说明为什么 size=1 的轴会造成传播冲突

- [x] **L3-11 `gather-scatter-batching`** — P1
  - 覆盖（6）：`explicit_gather_scatter_batching`
    + `executable_explicit_gather_scatter_batching/gather_iota_at_end_of_concat`
    + `executable_explicit_gather_scatter_batching/gather_iota_broadcast_concat`
    + `executable_explicit_gather_scatter_batching/gather_iota_concat_batch_dim`
    + `executable_explicit_gather_scatter_batching/gather_iota_reshaped_concat`
    + `executable_explicit_gather_scatter_batching/scatter_batch_dim`
  - 讲解：把 gather/scatter 的隐式批维转成显式批维，让分片规则能表达；
    5 个可执行用例覆盖 concat/iota/reshape 的组合
  - 验收：能解释隐式批维为什么会导致分片规则无法表达

---

### L4 — 导出核心（第 4 层 · 看补通信）

**层目标**：理解"传播结果 → 可执行 SPMD 程序"之间补了哪些通信，代价从哪来。
**覆盖**：`export/test/`(29) + `export/test/insert_explicit_reshards/`(29)
+ `export/test/optimize_collectives/`(2) + `export/test/resolve_permutation_factors/`(1) — **61 文件 / 16,271 行** ← 最大一层
**前置**：L1、L2

- [x] **L4-01 `export-pipeline-overview`** — P0
  - 覆盖（2）：`export_pipeline` `export_pipeline_explicit_collectives`
  - 讲解：导出流水线全貌与两条分支（是否显式插入 collective）；总览动画，后续每课回指本课
  - 验收：能画出导出流水线的 pass 顺序图

- [x] **L4-02 `reshard-insertion-overview`** — P0 ★ 本节总纲
  - 覆盖（2）：`insert_explicit_reshards` `insert_func_call_reshards`
  - 讲解：**为什么需要插入 reshard**：传播后某些算子仍"分片不兼容"，
    需要显式 reshard 让每个算子"对应维同分片、每个轴只切一种维度"；
    兼容性判定的三个条件与插入位置的选择策略；后续 L4-03～L4-07 按算子族展开
  - 验收：给定一段传播后的 IR，能圈出所有需要插 reshard 的位置

- [x] **L4-03 `reshard-elementwise-shape`** — P0
  - 覆盖（9）：`insert_explicit_reshards/elementwise_ops` `broadcast_in_dim` `bitcast_convert`
    `reshape` `reverse` `concatenate` `clamp_select` `pad_slice` `dynamic_slice_dynamic_update_slice`
  - 讲解：逐元素与形状变换算子的 reshard 规则；为什么 reshape 常需 reshard
  - 验收：给定算子两端不同分片，判断是否需要插 reshard

- [ ] **L4-04 `reshard-matmul-conv`** — P0
  - 覆盖（4）：`insert_explicit_reshards/dot_dot_general` `convolution` `fft` `cholesky_triangular_solve`
  - 讲解：矩阵/卷积类的分片冲突判定（非收缩维冲突、收缩维与复制轴互换）；插入位置选择
  - 验收：能解释文档中 `lhs {"x"}, rhs {"y"}` 例为何要在 rhs 前插 reshard

- [ ] **L4-05 `reshard-reduction`** — P1
  - 覆盖（4）：`insert_explicit_reshards/reduce` `reduce_window_select_and_scatter` `sort` `rng_bit_generator`
  - 讲解：归约类的 replica 组语义；sort 的被排序维必须全复制；RNG 的特殊约束
  - 验收：能说出 sort 为什么需要 need_replication 因子

- [ ] **L4-06 `reshard-memory-ops`** — P1
  - 覆盖（3）：`insert_explicit_reshards/gather_scatter` `custom_call` `collective_ops`
  - 讲解：访存类与自定义调用；集合通信算子自身的 reshard 处理
  - 验收：能判断 custom_call 的分片传播边界

- [ ] **L4-07 `reshard-structure`** — P1
  - 覆盖（9）：`insert_explicit_reshards/data_flow_ops` `manual_computation` `call_ops`
    `call_ops_enable_full_version_false` `func_inputs_outputs` `meshes`
    `single_device_sharding` `single_device_sharding_errors` `unreduced`
  - 讲解：数据流算子、函数边界、mesh 切换、单设备分片、未归约轴这些"结构性"场景的 reshard 插入
  - ✅ 含错误用例 `single_device_sharding_errors`
  - 验收：能解释函数边界为何是 reshard 高发区

- [ ] **L4-08 `reshard-to-collectives`** — P0 ★ 核心
  - 覆盖（2）：`reshard_to_collectives` `reshard_to_collectives_keep_redundant_reshards_true`
  - 讲解：单条 reshard 如何分解成 all-gather / all-reduce / all-slice / reduce-scatter / all-to-all /
    collective-permute 的组合；"每一维分片变化 → 一种通信"对照表动画
  - 验收：给定任意 reshard 前后分片，写出对应 collective 序列

- [ ] **L4-09 `resolve-permutation-factors`** — P1
  - 覆盖（2）：`resolve_permutation_factors`
    + `resolve_permutation_factors/resolve_permutation_factors_replica_id`
  - 讲解：permutation 因子（pad / reverse / 窗口类）的分片如何解析；halo exchange 开关差异；replica_id 场景
  - 验收：能说出开关 halo exchange 时 IR 的差别

- [ ] **L4-10 `optimize-collectives`** — P2
  - 覆盖（2）：`optimize_collectives/all_to_all_fully_scattered` `optimize_collectives/all_to_all_partially_scattered`
  - 讲解：消除 all-to-all 链前的冗余 collective-permute；全散开 vs 部分散开
  - 验收：能指出哪条 permute 被消除以及为什么安全

- [ ] **L4-11 `per-instruction-partitioning`** — P1
  - 覆盖（3）：`per_instruction_partitioning` `per_instruction_partitioning_range` `per_instruction_partitioning_subroutine`
  - 讲解：只对指定指令跑分区器并包进 `sdy.manual_computation`，用于 bisect；
    `filter` 语法（算子名子串 / `selectLow`/`selectHigh`/`func`）
  - 验收：能写出只分区 dot 与 pad 的 filter 串

- [ ] **L4-12 `export-named-computations`** — P2
  - 覆盖（1）：`export_named_computations`
  - 讲解：`named_computation` 反向 outline 成函数 + call，并保留分片
  - 验收：能预测 outline 后的函数签名

- [ ] **L4-13 `call-graph-flatten-unflatten`** — P2
  - 覆盖（2）：`unflatten_call_graph` `unflatten_call_graph_dedup_functions_fully_true`
  - 讲解：按 in/out 分片去重函数；`dedup-functions-fully` 的取舍
  - 验收：能判断两个函数在给定选项下是否会被合并

- [ ] **L4-14 `single-device-and-unreduced`** — P1
  - 覆盖（3）：`resolve_single_device_sharding` `verify_unreduced_axes` `remove_ag_rs_for_cmv1`
  - 讲解：单设备分片降级为 `stablehlo.if` 按设备号守卫；未归约轴一致性校验；CMV1 兼容性移除
  - 验收：能说出 unreduced 轴从哪来、必须在哪被消除

- [ ] **L4-15 `export-finalize`** — P0
  - 覆盖（7）：`close_shardings` `drop_sharding_rules` `drop_sharding_and_mesh` `inline_meshes`
    `remove_sharding_groups` `remove_sub_axes_in_input_output_shardings`
    `update_non_divisible_input_output_shardings`
  - 讲解：收尾清理各步目的与顺序；不可整除的输入输出分片如何被"截断"到可整除前缀；
    为什么要移除输入输出上的子轴
  - 验收：给一段 IR，说出该跑哪几个收尾 pass、顺序如何

- [ ] **L4-16 `sink-and-convert`** — P1
  - 覆盖（3）：`sink_data_flow_edges` `sink_func_data_flow_edges` `sharding_constraint_to_reshard`
  - 讲解：把边分片"下沉"到输入张量并删除边 op；constraint → reshard
  - 验收：能预测下沉后分片落在哪个值上

- [ ] **L4-17 `merger-and-debug-cleanup`** — P2
  - 覆盖（3）：`constant_or_scalar_merger` `remove_propagation_debug_info` `propagate_to_func_results`
  - 讲解：把分片相同的常量重新合并（与 L3-02 对称）；擦除调试属性；func 结果分片回填
  - 验收：能说明 L3-02 与 L4-17 为何是一对逆操作

---

### L5 — 全局 → 局部降级（第 5 层 · 看降级）

**层目标**：看懂 `tensor<8x8>` 如何变成每设备真实的 `tensor<4x4>`，以及每个算子如何降级。
**覆盖**：`export/test/convert_global_to_local/`(23) + `export/test/pad_for_divisibility/`(12) — 35 文件 / 4,240 行
**前置**：L4-08

- [ ] **L5-01 `global-to-local-overview`** — P0
  - 覆盖（2）：`convert_global_to_local/generic_ops` `convert_global_to_local/replica_id`
  - 讲解：类型转换器总体思路；`replica-count` / `partition-count` 作用；`enable-rgv3` / `per-dim-all-gather` 选项
  - 验收：能算出任意 sharding 对应的局部类型

- [ ] **L5-02 `lowering-sdy-collectives`** — P0
  - 覆盖（6）：`convert_global_to_local/sdy_all_gather` `sdy_all_reduce` `sdy_all_slice`
    `sdy_all_to_all` `sdy_reduce_scatter` `sdy_collective_permute`
  - 讲解：SDY 集合通信 → StableHLO 集合通信的映射；replica group 的构造
  - 验收：能写出某个 sdy.all_gather 对应的 stablehlo.all_gather 关键属性

- [ ] **L5-03 `lowering-sdy-structural`** — P1
  - 覆盖（3）：`convert_global_to_local/sdy_constant` `sdy_named_computation` `sdy_manual_computation`
  - 讲解：SDY 结构算子如何就地降级（常量按分片裁剪、named 内联、manual 直接展开为局部代码）
  - 验收：能预测 manual_computation 展开后的形状

- [ ] **L5-04 `lowering-elementwise-shape`** — P1
  - 覆盖（4）：`convert_global_to_local/stablehlo_concatenate` `stablehlo_iota` `stablehlo_slice` `stablehlo_pad`
  - 讲解：形状类算子在分片下的局部化；pad 的边界处理
  - 验收：能对某个 slice 分片场景算出局部 slice 参数

- [ ] **L5-05 `lowering-matmul`** — P0
  - 覆盖（2）：`convert_global_to_local/stablehlo_dot` `stablehlo_dot_general`
  - 讲解：matmul 在各种分片组合下的局部化 + 通信（经典并行策略的 IR 体现）
  - 验收：能说出"切 M / 切 N / 切 K"三种并行下各自的通信

- [ ] **L5-06 `lowering-convolution`** — P1
  - 覆盖（1）：`convert_global_to_local/stablehlo_convolution`
  - 讲解：卷积分片降级与 halo 处理
  - 验收：能指出卷积分片何时需要通信

- [ ] **L5-07 `lowering-reduction`** — P1
  - 覆盖（2）：`convert_global_to_local/stablehlo_reduce` `stablehlo_reduce_window`
  - 讲解：归约维分片时的局部归约 + all-reduce
  - 验收：能判断某个 reduce 是否需要通信

- [ ] **L5-08 `lowering-gather-scatter`** — P1
  - 覆盖（3）：`convert_global_to_local/stablehlo_gather` `stablehlo_scatter` `stablehlo_select_and_scatter`
  - 讲解：访存类算子的分片降级（与 L3-11、L6-05 呼应）
  - 验收：能说明 gather 的分片为何最难处理

- [ ] **L5-09 `pad-for-divisibility`** — P0
  - 覆盖（12）：`pad_for_divisibility/all_to_all` `dot_general` `func_ops` `generic_ops` `reduce_scatter`
    `sdy_all_slice_all_gather` `stablehlo_convolution` `stablehlo_gather` `stablehlo_pad`
    `stablehlo_reshape` `stablehlo_slice` `stablehlo_while`
  - 讲解：不可整除分片如何在导出时被补齐；padding 值选择；通信维度的 padding
  - 验收：给定 `tensor<7x3x8>` 沿 `z=3` 分片，写出补齐后形状与 pad 参数

---

### L6 — 可执行验证（第 6 层 · 看数值正确性）

**层目标**：理解 Shardy 如何**执行**分区结果证明其正确性，并能自己加测试。
**覆盖**：`export/test/executable_convert_global_to_local/`(39) + `export/test/executable_partitioner_pipeline/`(37) — 76 文件 / 4,062 行
**前置**：L4、L5

> **机制**（本层第 0 课必须讲清）：测试用 `split-file` 把 `.mlir` 拆成
> `part1`（带分片的全局程序）与 `part2`（手写的语义等价"参考实现"）；
> 脚本对 `part1` 跑完整分区流水线得到设备本地代码，拼接 `part2` 后用
> `stablehlo-translate --interpret` **实际执行**，要求两者结果一致。
> 即：**用数值一致性证明分区没有改变语义**。

- [ ] **L6-00 `executable-test-mechanism`** — P0
  - 覆盖：`executable_convert_global_to_local/run_sdy_interpreter_test.sh`
    `executable_partitioner_pipeline/run_sdy_interpreter_test.sh`（机制说明，非 IR）
  - 讲解：`split-file` / 流水线参数 / `--interpret` 全过程动画；如何自己写一个可执行测试
  - 验收：能独立写出一个最小可执行测试并通过

- [ ] **L6-01 `exec-sdy-collectives`** — P0
  - 覆盖（10）：`executable_convert_global_to_local/sdy_all_gather` `sdy_all_reduce` `sdy_all_slice`
    `sdy_all_to_all` `sdy_all_to_all_cross_replica` `sdy_collective_permute_cross_replica`
    `sdy_collective_permute_without_self_loops` `sdy_collective_permute_with_self_loops`
    `sdy_reduce_scatter` `sdy_constant`
  - 讲解：每段 IR 的 part1/part2 对照 + 逐设备数值动画（哪台设备算出什么）
  - 验收：能手工演算某个 4 设备 all-to-all 的数值结果

- [ ] **L6-02 `exec-stablehlo-collectives`** — P1
  - 覆盖（5）：`executable_convert_global_to_local/stablehlo_all_gather` `stablehlo_all_reduce`
    `stablehlo_all_to_all` `stablehlo_collective_permute` `stablehlo_reduce_scatter`
  - 讲解：直接写 stablehlo 集合通信时的等价性与差异
  - 验收：能说出与 L6-01 对应文件的异同

- [ ] **L6-03 `exec-convolution`** — P1
  - 覆盖（5）：`executable_convert_global_to_local/stablehlo_convolution_shard_batch`
    `stablehlo_convolution_shard_batch_group` `stablehlo_convolution_shard_contracting_dim`
    `stablehlo_convolution_shard_feature_group`
    + `executable_partitioner_pipeline/stablehlo_convolution_dual_semantics_factor_indivisible`
  - 讲解：卷积四种分片维度各自的正确性验证
  - 验收：能说出四种分片各自是否需要通信

- [ ] **L6-04 `exec-matmul-fft-iota`** — P1
  - 覆盖（4）：`executable_convert_global_to_local/stablehlo_dot_general` `stablehlo_fft` `stablehlo_iota`
    + `executable_partitioner_pipeline/stablehlo_dot_general_indivisible`
  - 讲解：matmul 数值等价；FFT 分片；iota 局部化
  - 验收：能解释 dot_general_indivisible 为何需要 padding

- [ ] **L6-05 `exec-gather`** — P0 ★ 最复杂
  - 覆盖（6）：`executable_convert_global_to_local/stablehlo_gather_shard_reduction_dim_is_collapsed`
    `stablehlo_gather_shard_reduction_dim_is_collapsed_i32`
    `stablehlo_gather_shard_reduction_dim_is_collapsed_max`
    `stablehlo_gather_shard_reduction_dim_is_collapsed_min`
    `stablehlo_gather_shard_reduction_dim_is_collapsed_not_in_start_index_map`
    `stablehlo_gather_shard_two_reduction_dims`
  - 讲解：gather 在归约维被折叠时的 4 种变体（含 i32/max/min 三种归约语义）；两个归约维
  - 验收：能区分 4 个变体差异并预测数值

- [ ] **L6-06 `exec-pad`** — P0 ★ 本层最大族
  - 覆盖（19）：`executable_convert_global_to_local/stablehlo_pad_non_sharded`
    `stablehlo_pad_sharded_non_uniform` `stablehlo_pad_sharded_uniform`
    + `executable_partitioner_pipeline/stablehlo_pad_indivisible` `stablehlo_pad_interior`
    `stablehlo_pad_large_pad` `stablehlo_pad_large_pad_within_one_hop` `stablehlo_pad_left_shift`
    `stablehlo_pad_multidim_mixed_shifts` `stablehlo_pad_multidim_with_hops`
    `stablehlo_pad_multiple_hops_right_shift` `stablehlo_pad_multiple_right_hops`
    `stablehlo_pad_replica_id` `stablehlo_pad_replicated_dual_slice_pad`
    `stablehlo_pad_replicated_negative_high_padding` `stablehlo_pad_replicated_negative_low_padding`
    `stablehlo_pad_right_shift` `stablehlo_pad_single_left_hop` `stablehlo_pad_single_right_hop`
  - 讲解：padding 的分片与 **halo exchange**（多跳邻居数据交换）；"数据移动路径"动画
  - 验收：能画出某个 left_shift 场景下每台设备需要向谁要哪些数据

- [ ] **L6-07 `exec-reshape`** — P0
  - 覆盖（7）：`executable_partitioner_pipeline/stablehlo_reshape_1d_to_2d_split`
    `stablehlo_reshape_1d_to_2d_split_2groups`
    `stablehlo_reshape_1d_to_2d_split_custom_device_ids`
    `stablehlo_reshape_1d_to_2d_split_gap_2`
    `stablehlo_reshape_1d_to_2d_split_passthrough_indivisible`
    `stablehlo_reshape_2d_split_unrelated_axis` `stablehlo_reshape_axis_shift`
  - 讲解：reshape 的轴拆分（与 `intro` 第 9 幕子轴概念直接对应，这里看真实降级结果）
  - 验收：能给 1D→2D reshape 选出正确的子轴拆分方案

- [ ] **L6-08 `exec-reverse-slice`** — P1
  - 覆盖（9）：`executable_partitioner_pipeline/stablehlo_reverse_multi_dim_divisible`
    `stablehlo_reverse_single_dim_indivisible` `stablehlo_slice_comm_free` `stablehlo_slice_indivisible`
    `stablehlo_slice_replicated` `stablehlo_slice_replicated_mesh_2` `stablehlo_slice_strided`
    `stablehlo_slice_with_communication`
    + `executable_convert_global_to_local/stablehlo_slice`
  - 讲解：reverse 的分片；slice 在"免通信 / 需通信"之间的分界
  - 验收：能判断某个 slice 是否需要通信

- [ ] **L6-09 `exec-scatter-and-misc` — P1
  - 覆盖（7）：`executable_convert_global_to_local/stablehlo_scatter_replicated_bounds`
    `stablehlo_scatter_shard_implicit_dim` `stablehlo_scatter_shard_indexed_inserted_dim`
    `stablehlo_select_and_scatter` `stablehlo_concatenate` `stablehlo_reduce_window` `stablehlo_sort`
  - 讲解：scatter 三种分片形态；其余算子收尾
  - 验收：能解释 scatter 隐式维与显式维分片的区别

- [ ] **L6-10 `exec-single-device-and-alltoall`** — P2
  - 覆盖（4）：`executable_partitioner_pipeline/single_device_add` `single_device_switch`
    `sdy_all_to_all_fully_scattered` `sdy_all_to_all_partially_scattered`
  - 讲解：单设备守卫的数值验证；all-to-all 优化的端到端验证
  - 验收：能说明单设备分片如何被执行

---

### L7 — 综合实战（第 7 层）

**层目标**：把前六层串起来，面对真实模型能独立分析。
**覆盖**：不新增测试文件，**复用**前述各课 IR 做综合。

- [ ] **L7-01 `end-to-end-walkthrough`** — P0
  - 讲解：一个完整 StableHLO 程序，从导入到设备代码，**一条时间轴走完所有 pass**，每步可暂停看 IR
  - 验收：能对着动画复述完整流程

- [ ] **L7-02 `parallelism-strategies`** — P0
  - 讲解：用同一份程序演示数据并行 / 张量并行 / 流水线并行 / ZeRO 的分片标注差异与通信代价对比
  - 验收：给定模型规模与设备数，能选合理策略并说明代价

- [ ] **L7-03 `debugging-playbook`** — P0
  - 讲解：常见问题排查：分片没传播过去 / 出现意外的 all-gather / 不可整除报错 /
    `debug-sharding-origins` 用法 / `module-dump-directory` 用法
  - 验收：给定一个"坏"IR 能定位原因

- [ ] **L7-04 `dialect-agnostic-integration`** — P1
  - 讲解：为自己的方言接入 Shardy：实现 `ShardingRuleOpInterface` / `ShardableDataFlowOpInterface`、
    常量拆分与 Elementwise trait
  - 验收：能写出一条自定义算子的 sharding rule 并被传播正确使用

---

## 3. 覆盖度矩阵（241 文件 → 课件）

> **验收硬指标**：下表每行的文件必须能在对应课件的 `source.md` 中找到逐字引用。

| 层 | 测试目录 | 文件数 | 行数 | 对应课件 |
|---|---|---|---|---|
| L1 | `ir/test/` | 28 | 5,048 | L1-01 ～ L1-10 |
| L2 | `transforms/propagation/test/` | 16 | 7,684 | L2-01 ～ L2-11 |
| L2 | `transforms/propagation/debugging/test/` | 2 | 1,160 | L2-12 |
| L3 | `transforms/import/test/` | 18 | 6,499 | L3-01 ～ L3-10 |
| L3 | `transforms/import/test/executable_explicit_gather_scatter_batching/` | 5 | 213 | L3-11 |
| L4 | `transforms/export/test/` | 29 | 11,349 | L4-01、L4-02、L4-08 ～ L4-17 |
| L4 | `transforms/export/test/insert_explicit_reshards/` | 29 | 4,669 | L4-03 ～ L4-07 |
| L4 | `transforms/export/test/optimize_collectives/` | 2 | 231 | L4-10 |
| L4 | `transforms/export/test/resolve_permutation_factors/` | 1 | 22 | L4-09 |
| L5 | `transforms/export/test/convert_global_to_local/` | 23 | 2,791 | L5-01 ～ L5-08 |
| L5 | `transforms/export/test/pad_for_divisibility/` | 12 | 1,449 | L5-09 |
| L6 | `transforms/export/test/executable_convert_global_to_local/` | 39 | 2,163 | L6-01 ～ L6-09 |
| L6 | `transforms/export/test/executable_partitioner_pipeline/` | 37 | 1,899 | L6-03 ～ L6-10 |
| | **合计** | **241** | **45,177** | **7 层 / 74 课** |

---

## 4. 里程碑与工作量

| 里程碑 | 内容 | 课件数 | 覆盖文件 | 依赖 |
|---|---|---|---|---|
| **M0** | 提炼 `shared/` 引擎 + `tools/` 校验脚本 + `README.md` 索引 | — | 0 | `intro/` 已稳定 |
| **M1** | L1 全部（IR 构件） | 10 | 28 | M0 |
| **M2** | L2 全部（传播算法） | 12 | 18 | M1 |
| **M3** | L3 全部（导入流水线） | 11 | 23 | M2 |
| **M4** | L4 全部（导出核心） | 17 | 61 | M2 |
| **M5** | L5 全部（降级） | 9 | 35 | M4 |
| **M6** | L6 全部（可执行验证） | 11 | 76 | M5 |
| **M7** | L7 全部（综合）+ 全量验收 | 4 | 0（复用） | M6 |

**建议顺序**：M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7。
M3 与 M4 可并行（都只依赖 M2）。

**每课工作量参考**：约 0.5～2 人日。
大文件课（L3-02 1782 行、L4-13 1646 行、L4-09 1463 行、L4-11 1263 行、L2-10 1213 行、
L2-11 1200 行、L2-01 1106 行）需拆成**多条动画线**，取上限。

---

## 5. 验收计划

### A. 内容准确性（可自动断言）

- [ ] **A1 · IR 保真**：`tools/check_ir_fidelity.py`
  每课 `source.md` 标注的每个 IR 片段，必须是源测试文件中的**逐字子串**
  （比对前统一去除 `// CHECK` 行与行首缩进）。失败即报错，防止"记忆式改写"引入错误语法。
- [ ] **A2 · flag 正确**：`tools/check_flags.py`
  课件中每个 `-sdy-*` 必须以 pass 名或 `let option` 出现在对应 `passes.td` 中。
- [ ] **A3 · 清单一致**：每课 `README.md` 的"覆盖文件"清单与 `source.md` 实际引用集合完全相同。

### B. 覆盖度（硬指标）

- [ ] **B1 · 全覆盖**：`tools/check_coverage.py`
  遍历 `shardy/dialect/sdy/**/test/**/*.mlir` 得 241 个路径，
  与所有课件 `source.md` 引用的并集求差 —— **差集必须为空**。
- [ ] **B2 · 无空课**：每课至少引用 1 个测试文件。
- [ ] **B3 · 无重复遗漏**：每个文件至少被 1 课引用（允许多课引用，但需在 §3 矩阵中声明主课）。
- [ ] **B4 · 计数断言**：总文件数 = 241、总行数 = 45,177；数值变化时报警提示上游更新。

### C. 渲染质量（复用 `intro` 已验证方法）

- [ ] **C1 · 无 JS 错误**：`tools/check_render.py` 用无头 Chrome 打开每课，
  捕获 `pageerror` 与 `console.error` —— 必须为 0。
- [ ] **C2 · 无布局溢出**：每课在早/中/末三个时间点快照，遍历可视区元素比对边界 —— 溢出数必须为 0。
  （方法已在 `intro` 验证：借此查出 ID 冲突、flex `min-width:auto` 撑宽两个真实 bug。）
- [ ] **C3 · 多分辨率**：1280×720 与 1920×1080 下均通过 C2。
- [ ] **C4 · 交互自检**：播放/暂停/上一课/下一课/方向键/圆点跳转全部生效。
- [ ] **C5 · 离线可用**：断网打开仍完整渲染（无 CDN）；资源加载无 404（含 favicon）。

### D. 教学效果

- [ ] **D1 · 三段齐全**：每课都有"看 IR / 看动画 / 做练习"三段。
- [ ] **D2 · 前置链完整**：每课 `README.md` 声明前置课，且前置课编号小于本课（无环）。
- [ ] **D3 · 术语一致**：全站统一术语表（`animation/README.md` 维护）；
  同一概念在 `intro` 与各层课件中译名一致（factor=因子、dim sharding=维分片、…）。
- [ ] **D4 · 难度单调**：脚本检查"概念首次出现课号 ≤ 其定义课号"，防止 L1 用到 L4 才讲的概念。

### E. 可维护性

- [ ] **E1 · 引擎单一来源**：`shared/` 是唯一实现；各课件不得内联复制引擎代码。
- [ ] **E2 · 独立可开**：任一课件 `index.html` 双击可独立打开。
- [ ] **E3 · 回归**：改动 `shared/` 后，C 组必须对全部课件重跑通过。

### 验收命令（M0 完成后即可用）

```sh
cd animation
python3 tools/check_coverage.py        # B 组
python3 tools/check_ir_fidelity.py     # A1/A3
python3 tools/check_flags.py           # A2
python3 tools/check_render.py --all    # C 组
```

**全部通过 = 计划完成。**
C 组脚本可直接从 `intro` 的验证脚本改造（已在 `intro` 上跑通：20/20 幕渲染、0 溢出、0 错误）。

---

## 6. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| L4/L6 文件多且碎（61 + 76） | 工期长、易漏 | 按算子族**合并成课**（如 L6-06 pad 一族 19 文件一课），用 §3 矩阵 + B1 脚本兜底 |
| 上游测试 IR 变动 | 课件与源脱节 | A1/A3 脚本化；B4 在行数变化时报警，提示同步 |
| 测试含 FileCheck 变量（`%[[X]]`） | 误当字面 IR 展示 | 展示时统一去除 `CHECK` 行，只讲 `RUN` + 输入 IR + 关键输出；`source.md` 保留原文便于核对 |
| 大文件（1782 / 1665 / 1646 / 1463 / 1263 / 1213 / 1200 / 1106 行） | 单课过载 | 按**语义分组**切成多条动画线，一课内可翻页；不追求一屏讲完 |
| 可执行测试需构建产物 | 无法演示真实执行 | L6 用 `split-file` 后的 part1/part2 做**静态数值推演动画**，并给出本地复跑命令 |

---

## 7. 明确不做的部分（范围边界）

1. **`*_test.cc` C++ 单测** —— 测的是 C++ API 而非 IR，不在"IR 动画化"范围内：
   `ir/axis_list_ref_test.cc` `ir/dialect_test.cc` `ir/utils_test.cc`
   `transforms/export/utils_test.cc`
   `transforms/propagation/aggressive_factor_propagation_test.cc`
   `transforms/propagation/auto_partitioner_registry_test.cc`
   `transforms/propagation/sharding_projection_test.cc`
   `transforms/propagation/basic_factor_propagation_test.cc`
   `transforms/common/op_properties_test.cc`
2. **`mpmd` 方言**的测试 IR —— 本计划范围限定 `shardy/dialect/sdy`。
   MPMD 已在 `intro` 第 19 幕概览；如需深入应另立 `animation/mpmd/` 计划。
3. **`stablehlo` / `llvm` 上游测试**。
