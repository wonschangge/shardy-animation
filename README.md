# Shardy 动画课件

把 [openxla/shardy](https://github.com/openxla/shardy) 的 **SDY 方言**从语法讲到分区器产出的设备代码。
共 7 层 / 74 课，覆盖 `shardy/dialect/sdy` 下**全部 241 个测试 IR 文件**（45,177 行）。

## 怎么打开

**方式一（推荐）**：浏览器直接打开任意课件目录下的 `index.html`。
纯静态、无 CDN、无构建步骤，**离线可用，双击即可**。

**方式二**：起一个本地静态服务（便于多课件间跳转）：

```sh
cd animation
python3 -m http.server 8801 --bind 0.0.0.0
# 然后访问 http://<本机IP>:8801/intro/
```

**操作**：`空格` 暂停/播放（真正冻结幕内动画），`←` `→` 上一幕/下一幕，`Home` 回到开头，底部圆点直接跳转。

---

## 学习路线

| 层 | 目录 | 认知目标 | 看完能… | 课数 | 覆盖文件 |
|---|---|---|---|---|---|
| 预备 | `intro/` | 建立直觉 | 知道 Shardy 是什么、分片怎么表示 | 20 幕 | — |
| L1 | `L1-ir/` | **认识构件** | 读懂任何 SDY IR，知道每个 op/attr 是什么 | 10 | 28 |
| L2 | `L2-propagation/` | **理解算法** | 预测传播结果，看懂调试输出 | 12 | 18 |
| L3 | `L3-import/` | **理解准备** | 知道传播前做了哪些规范化，为什么 | 11 | 23 |
| L4 | `L4-export-core/` | **理解代价** | 知道通信从哪来、reshard 怎么变 collective | 17 | 61 |
| L5 | `L5-lowering/` | **理解降级** | 看懂 `tensor<8x8>` 怎么变成 `tensor<4x4>` | 9 | 35 |
| L6 | `L6-executable/` | **验证正确** | 能自己加一个可执行测试验证数值正确 | 11 | 76 |
| L7 | `L7-capstone/` | **综合运用** | 独立分析真实模型的并行策略 | 4 | 复用 |

完整规划（每课覆盖哪些文件、讲什么、验收点）见 **[TODOLIST.md](TODOLIST.md)**。
执行约束（Definition of Done、门禁、git 规范）见 **[AGENTS.md](AGENTS.md)**。

---

## 目录结构

```
animation/
├── README.md          # 本文件
├── TODOLIST.md        # 计划：7 层 / 74 课，逐文件覆盖矩阵
├── AGENTS.md          # 执行约束：DoD / 门禁 / 提交与推送规范
├── shared/            # 共享动画引擎（单一来源，改这里影响所有课件）
│   ├── engine.js      #   U（底层元件）/ Timeline（时间轴）/ App（播放器）
│   ├── widgets.js     #   W（课件级元件：IR 对照框、步骤器、练习题、表格…）
│   ├── theme.css      #   主题、版式、动画
│   └── lesson-shell.js#   页面骨架生成 + SHELL.boot()
├── tools/             # 验收门禁
│   ├── check_coverage.py      # B 组：覆盖度
│   ├── check_ir_fidelity.py   # A1/A3：IR 逐字保真
│   ├── check_flags.py         # A2：pass flag 真实性
│   └── check_render.py        # C 组：渲染质量
├── intro/             # 预备篇（20 幕，已完成）
└── L1-ir/ … L7-capstone/   # 各层课件（随计划落地）
```

---

## 术语表（全站统一口径）

翻译以本表为准，`intro/` 与各层课件必须一致。

### 分片表示

| 英文 | 中文 | 说明 |
|---|---|---|
| logical mesh / mesh | 逻辑网格 / 网格 | 设备的逻辑多维视图，由若干命名轴构成 |
| axis | 轴 | 网格的一个维度，有名字和大小 |
| device | 设备 | 参与计算的硬件单元 |
| tensor sharding | 张量分片 | 描述张量如何分布到设备上 |
| dim sharding | 维分片 | 张量某一维沿哪些轴切分，major→minor |
| open dimension | 开维 | 标 `?`，传播可继续往上加轴 |
| closed dimension | 闭维 | 不标 `?`，锁定，传播不可改 |
| explicit replication | 显式复制 | `replicated={...}`，锁死该轴不参与分片 |
| implicit replication | 隐式复制 | 该轴未被任何维使用，即复制 |
| sub-axis / axis splitting | 子轴 / 轴拆分 | 把一个轴拆成多个子轴，记 `"x":(pre)size` |
| priority | 优先级 | `p0/p1/p2`，决定传播轮次 |
| unreduced axis | 未归约轴 | 该轴上只有部分和，需 all-reduce |
| divisibility | 整除性 | 维大小能否被分片轴大小整除；不整除需 padding |
| local shape | 局部形状 | 单台设备上真实持有的形状 |
| global shape | 全局形状 | 逻辑上的完整形状 |

### 传播

| 英文 | 中文 | 说明 |
|---|---|---|
| propagation | 传播 | 从已知分片推导全图分片的过程 |
| factor | 因子 | 传播的内部抽象，维度由因子构成 |
| op sharding rule | 算子分片规则 | 用 einsum 风格描述算子各维与因子的映射 |
| pass-through factor | 直通因子 | 同因子在所有张量上应同样分片 |
| reduction factor | 归约因子 | 如 matmul 的收缩维，结果是部分和 |
| need_replication factor | 必须复制因子 | 该因子被分片时需全复制，如 sort 的被排序维 |
| permutation factor | 置换因子 | 需 collective-permute 或 halo exchange |
| blocked propagation factor | 阻断传播因子 | 禁止沿该因子传播 |
| fixed point | 不动点 | 传播迭代到不再变化的状态 |
| conservative propagation | 保守传播 | 禁止分裂轴与不可整除轴 |
| aggressive propagation | 激进传播 | 主动消解冲突，可能引入通信 |
| op priority | 算子优先级 | 按算子类型分批传播 |
| user priority | 用户优先级 | 按 `p0/p1/p2` 分轮传播 |
| conflict resolution | 冲突消解 | 分片要求不一致时的裁决机制 |
| data flow edge | 数据流边 | source/target 必须同分片的桥接关系（while/call 等） |
| sharding origin | 分片来源 | 调试信息：某分片由谁引入 |

### 编译器 API

| 英文 | 中文 | 说明 |
|---|---|---|
| in/out sharding | 输入/输出分片 | 标在函数参数/返回值上 |
| sharding constraint | 分片约束 | `sdy.sharding_constraint`，钉住中间张量 |
| sharding group | 分片组 | `sdy.sharding_group`，强制同组同分片 |
| manual computation | 手动计算 | `sdy.manual_computation`，手写局部代码与通信 |
| named computation | 命名计算 | `sdy.named_computation`，让传播穿过函数调用 |
| propagation barrier | 传播屏障 | `sdy.propagation_barrier`，限制传播方向 |
| reshard | 重分片 | `sdy.reshard`，声明需要换分片 |

### 降级与执行

| 英文 | 中文 | 说明 |
|---|---|---|
| collective | 集合通信 | 跨设备数据移动算子 |
| all-gather | 全收集 | 分片 → 复制 |
| all-reduce | 全归约 | 部分和 → 全和（复制） |
| all-slice | 全切片 | 复制 → 分片 |
| reduce-scatter | 归约散射 | 先归约再分片 |
| all-to-all | 全交换 | 设备间互换切片 |
| collective-permute | 集合置换 | 点对点重排/替换轴 |
| halo exchange | 晕轮交换 | 窗口类算子的邻居数据交换 |
| padding | 补齐 | 让不可整除的分片变得可整除 |
| partitioner | 分区器 | 把全局程序切成设备本地程序的组件 |
| SPMD | 单程序多数据 | 同一份程序跑在所有设备上，数据不同 |
| replica / partition | 副本 / 分区 | 数据并行维度 / 模型并行维度 |
| MPMD | 多程序多数据 | 不同设备组跑不同程序片段 |
| fragment | 片段 | MPMD 中绑定到某网格的一段计算 |
| transfer | 搬运 | MPMD 中跨网格传数据 |

---

## 课件索引

> 随课件落地更新。每课目录含 `index.html` / `lesson.js` / `source.md` / `README.md`。

| 课件 | 主题 | 覆盖文件 | 状态 |
|---|---|---|---|
| [`intro/`](intro/) | 预备篇：分片表示 / 传播 / 集合通信 / SPMD / MPMD 概览（20 幕） | — | ✅ |
| [`L1-01`](L1-ir/L1-01-mesh-and-devices/) | 逻辑网格 `sdy.mesh` 与设备编号（7 幕） | 2 | ✅ |
| [`L1-02`](L1-ir/L1-02-tensor-sharding-syntax/) | 张量分片属性语法 `#sdy.sharding`（10 幕） | 2 | ✅ |
| [`L1-03`](L1-ir/L1-03-tensor-sharding-verification/) | 分片校验不变量（33 类错误 / 9 幕） | 1 | ✅ |
| [`L1-04`](L1-ir/L1-04-edge-sharding/) | 边分片与传播调试信息 `propagation_edges`（7 幕） | 2 | ✅ |
| [`L1-05`](L1-ir/L1-05-op-sharding-rule/) | 算子分片规则 `op_sharding_rule`（9 幕） | 3 | ✅ |
| [`L1-06`](L1-ir/L1-06-collectives/) | 集合通信算子（8 个 / 10 幕） | 3 | ✅ |
| [`L1-07`](L1-ir/L1-07-manual-computation/) | 手动计算 `manual_computation`（8 幕） | 3 | ✅ |
| [`L1-08`](L1-ir/L1-08-named-computation-and-dataflow/) | 命名计算与数据流边（7 幕） | 4 | ✅ |
| [`L1-09`](L1-ir/L1-09-constraint-group-barrier/) | 约束类算子 constraint/group/barrier（7 幕） | 4 | ✅ |
| [`L1-10`](L1-ir/L1-10-reshard-and-constant/) | 重分片与常量 `reshard`/`constant`（8 幕） | 4 | ✅ |
| — | **L1 层完成**：10 课 / 28 文件 | 28 | ✅ |
| L2-01 | 见 [TODOLIST.md §2](TODOLIST.md) —— 下一层：传播算法 | | ⬜ |

---

## 验收门禁

改完任何课件，提交前必须四组全绿：

```sh
cd animation
python3 tools/check_coverage.py                  # B 组：覆盖度（最终应 241/241）
python3 tools/check_ir_fidelity.py               # A1/A3：IR 逐字来自源测试文件
python3 tools/check_flags.py                     # A2：pass flag 真实存在
python3 tools/check_render.py --all              # C 组：渲染 / 溢出 / 交互
python3 tools/check_render.py --all --res 1280x720 1920x1080   # 多分辨率
```

| 脚本 | 断言 |
|---|---|
| `check_coverage.py` | 上游全部测试 `.mlir` 都被课件 `source.md` 声明；无空课；无幻影路径 |
| `check_ir_fidelity.py` | 课件中每个 ```mlir 代码块都能在声明的源文件里找到**连续逐字片段**（省略处用单独一行 `...` 分段） |
| `check_flags.py` | 每个 `-sdy-*` / `-mpmd-*` 及 pass 选项都能在 `sdy_opt/mpmd_opt --help` 中找到 |
| `check_render.py` | 无 JS 错误、无布局溢出、交互可用、资源无 404 |

---

## 新增一课

1. 在 `TODOLIST.md` 找到该课的「覆盖」清单，**逐文件**读上游测试 IR。

   > **排版基准**：`#visual` 可视区固定为 **780 × 521 舞台像素**。
   > 卡片等固定宽度元素要按这个尺寸算，一行放几个必须满足 `n×w + (n-1)×gap ≤ 780`。
   > 无头浏览器量到的 `getBoundingClientRect` 是屏幕像素，**必须除以缩放比**（1440×900 下为 1.125）再判断。
2. 建目录 `L<层>/L<层>-<序号>-<slug>/`，写四个文件：

   ```html
   <!-- index.html 全文就这么长 -->
   <!DOCTYPE html><html lang="zh-CN"><head>
   <meta charset="utf-8"><title>…</title>
   <link rel="icon" href="…">
   <link rel="stylesheet" href="../../shared/theme.css">
   </head><body>
   <script src="../../shared/engine.js"></script>
   <script src="../../shared/widgets.js"></script>
   <script src="../../shared/lesson-shell.js"></script>
   <script src="lesson.js"></script>
   <script>SHELL.boot(SCENES, {kicker:'L1 · IR 构件', codeCap:'对应 IR'});</script>
   </body></html>
   ```

3. `source.md` 用机器可读块声明覆盖范围（门禁据此断言）：

   ```markdown
   <!-- sdy-coverage
   ir/test/mesh_parse_print.mlir
   ir/test/mesh_verification.mlir
   -->

   ## 片段 1：命名轴
   ```mlir
   sdy.mesh @mesh_1d = <["a"=4]>
   ```
   ```

   路径相对于 `shardy/dialect/sdy/`。IR 必须**逐字**，需省略处用单独一行 `...`。

4. `lesson.js` 定义 `SCENES` 数组（结构与 `intro/scenes.js` 相同）：

   ```js
   const SCENES = [{
     kicker:'L1 · IR 构件', title:'…', sub:'…', caption:'…',
     code:`…`,                       // 侧栏 IR
     duration: 12000,                // 自动播放时长(ms)
     build(root, tl) {               // root: 可视区容器; tl: 时间轴
       tl.at(900, () => { /* 900ms 时执行一次 */ });
     }
   }];
   ```

5. 跑四个门禁 → 全绿 → 按 `AGENTS.md` §5.2 的格式提交并推送。

---

## 贡献约定

- **引擎单一来源**：所有课件引用 `shared/`，不得内联复制引擎代码；改 `shared/` 后必须全量回归。
- **IR 逐字**：不得凭记忆改写测试 IR；展示时去掉 `// CHECK` 行，`source.md` 保留原文。
- **离线可用**：不引入任何 CDN 或外部依赖。
- 详细约束见 **[AGENTS.md](AGENTS.md)**。
