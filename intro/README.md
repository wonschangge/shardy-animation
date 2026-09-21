# Shardy 入门动画

面向新手的 Shardy 特性动画讲解，共 **20 幕**，覆盖从"为什么需要切分"到 MPMD 的完整特性链路。

## 如何观看

**方式一（推荐）**：直接用浏览器打开 `index.html`（纯静态，无需构建、无外部依赖、可离线）。

**方式二**：本机已启动一个静态服务：

```
http://192.168.10.20:8801/
```

## 操作

| 操作 | 按键 |
|---|---|
| 播放 / 暂停 | `空格`（暂停会冻结幕内分步动画，不只是停止翻页） |
| 上一幕 / 下一幕 | `←` / `→` |
| 回到开头 | `Home` 或 `⏮` |
| 跳到任意幕 | 点击底部圆点 |

自动播放会按每幕时长推进，底部进度条显示当前幕进度。

## 20 幕内容

| # | 主题 | 关键点 |
|---|---|---|
| 1 | 封面 | Shardy 是什么 |
| 2 | 为什么切分 | 复制 vs 切分，显存与可扩展性 |
| 3 | `sdy.mesh` | 逻辑网格、轴命名、device_ids |
| 4 | `#sdy.sharding` ★ | 轴基分片表示，张量如何落到设备 |
| 5 | 局部形状 | `局部维 = 全局维 ÷ ∏轴大小`，不可整除分片 |
| 6 | 隐式复制 | 没被用到的轴 = 复制 |
| 7 | 开维 / 闭维 | `{"x"}` vs `{"x", ?}` |
| 8 | 显式复制 | `replicated={"y"}` 锁死轴 |
| 9 | 子轴与轴拆分 | `"x":(1)2`，reshape 零通信 |
| 10 | 优先级 | `p0/p1/p2` 分轮传播 |
| 11 | 算子分片规则 | einsum 因子映射、reduction 因子导致 all-reduce |
| 12 | 传播算法 | 沿 factor 传播、不动点迭代 |
| 13 | 数据流算子 | `while` / `named_computation` / `manual_computation` 的专用机制 |
| 14 | 冲突消解层级 | 基础 → 激进 → 算子优先级 → 用户优先级 |
| 15 | 编译器 API | 输入输出分片、`sharding_constraint`、`sharding_group`、`manual_computation`、`propagation_barrier` |
| 16 | 三条流水线 | 导入 / 传播 / 导出及各段 pass |
| 17 | 六种集合通信 | all-gather / all-reduce / all-slice / reduce-scatter / all-to-all / collective-permute |
| 18 | SPMD 落地 | 全局形状 → 局部形状 |
| 19 | MPMD | topology / fragment / stage / transfer |
| 20 | 速查表 | 语法与上手命令 |

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 页面外壳 |
| `styles.css` | 样式与动画 |
| `app.js` | 舞台缩放、时间轴引擎、可视元件（设备网格 / 张量网格 / 高亮） |
| `scenes.js` | 20 幕的全部内容与分步动画 |

## 关于准确性

- 动画中的每一段 IR 都取自 `openxla/shardy` 仓库的 `docs/` 与测试用例（`shardy/dialect/**/test/*.mlir`），
  经逐条核对，未使用杜撰语法。
- 命令行 flag 名称核对自各 `passes.td`。
- 一个易错点已在动画中修正：**不存在 `mpmd.mesh` 这个 op** —— mesh 只存在于
  `#mpmd.topology<...>` 属性和 `!mpmd.mesh_tensor<...>` 类型中。

## 修改

改 `scenes.js` 里的 `SCENES` 数组即可。每一幕的结构：

```js
{
  kicker, title, sub,          // 顶部标题区（支持 HTML）
  caption,                     // 底部说明
  code, codeCap,               // 侧栏 IR
  duration,                    // 自动播放时长(ms)
  build(root, tl) {            // root: 可视区容器; tl: 时间轴
    tl.at(1200, () => { ... });  // 1200ms 时执行一次
    tl.every(2000, () => { ... }); // 每 2000ms 执行
  }
}
```

时间轴只在"播放中"前进，所以 `空格` 暂停是真正的时间冻结。
