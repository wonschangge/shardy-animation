# AGENTS.md — 执行 `TODOLIST.md` 的约束条件

> 本文件是**执行 `animation/TODOLIST.md` 时必须遵守的硬约束**。
> `TODOLIST.md` 定义**做什么**（7 层 / 74 课 / 覆盖 241 个测试 IR）；本文件定义**怎么做才算合格**。
>
> 冲突时以本文件为准；本文件未覆盖的，以 `TODOLIST.md` 为准。

---

## 0. 一句话要求

**保质保量**：74 课一课不少、241 个测试文件一个不漏、每课都必须通过 A/B/C 组验收；
**每次变更都要有 git 提交信息，并且推送到远端。**

---

## 1. 环境事实（已实测，直接使用，勿再试探）

| 项 | 值 |
|---|---|
| 仓库根 | `/data/WORKSPACE/shardy-project/animation`（**注意：不是 shardy 上游仓库**） |
| 远端 | `https://github.com/wonschangge/shardy-animation.git`（**private**） |
| 分支 | `main`（默认分支，唯一长期分支） |
| 上游参考代码 | `/data/WORKSPACE/shardy-project/`（openxla/shardy 克隆，**只读**，见 §7） |

### 1.1 git 身份与凭据（**都是仓库级配置，不得改成全局**）

```sh
user.name  = Chanj Wons
user.email = 892108131@qq.com
credential.https://github.com.helper = !gh auth git-credential
http.proxy = http://127.0.0.1:7890
```

- `gh` 已登录为 `wonschangge`，具备 `repo` scope，可直接建仓/推送。
- **不要**运行 `gh auth setup-git`（那会写全局配置）。
- 首次执行前自检：`cd animation && git config --local --list`。若身份/凭据缺失，按上表用
  `git config --local` 补回，**不要**用 `--global`。

### 1.2 网络

- 外网必须经代理 `http://127.0.0.1:7890`。本仓库已设 `http.proxy`；
  **`gh` 命令需自行导出环境变量**：

  ```sh
  export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
  ```

- 推送失败先查代理，再查凭据，不要盲目重试。

### 1.3 磁盘

- 根分区 `/`（含 `/home`、`/tmp`）**长期接近满载**。
- 所有产物、临时文件、截图、验证脚本输出一律写 `/data/...`，**不得写 `/tmp` 或 `$HOME`**。

### 1.4 验证工具（已就绪）

| 用途 | 方式 |
|---|---|
| 无头浏览器 | `python3` + `playwright`，`launch(executable_path="/usr/bin/google-chrome")` |
| Node 语法检查 | `node --check <file>.js` |
| 本地静态服务 | `python3 -m http.server <port> --bind 0.0.0.0`（后台任务） |
| 复跑 SDY pass | `/data/WORKSPACE/shardy-project/bazel-bin/shardy/tools/sdy_opt` |
| 重新构建（必要时） | `/data/bin/bzl build -c opt //shardy/...`（**勿用裸 `bazel`**） |

---

## 2. 开工前置（M0，**必须先于任何课件**）

验收门禁脚本必须先存在，否则后续每课的"通过验收"无法证明。M0 必须在**单独一次提交**中完成：

- [ ] `shared/`：从 `intro/app.js` + `intro/styles.css` 提炼引擎（`engine.js` / `widgets.js` / `theme.css` / `lesson-shell.js`）。
  **提炼后必须回归验证 `intro/` 仍 20/20 渲染、0 溢出**，否则不得继续。
- [ ] `tools/check_coverage.py`、`tools/check_ir_fidelity.py`、`tools/check_flags.py`、`tools/check_render.py`
- [ ] `animation/README.md`：学习路线图 + 课件索引 + 术语表
- [ ] 四个脚本能在**空课件集**上运行且不误报（覆盖率应为 0/241）

> M0 未完成就写课件 = 违反本约束。

---

## 3. 质量约束（保质）

### 3.1 每课的 Definition of Done

一课**只有全部满足**才允许提交：

- [ ] 目录结构符合 `TODOLIST.md` §1.2：`index.html` / `lesson.js` / `source.md` / `README.md`
- [ ] `index.html` **双击可直接打开**：无 CDN、无构建步骤、离线可用
- [ ] 三段结构齐全：**看 IR** → **看动画** → **做练习**
- [ ] `source.md` 中引用的 IR 片段**逐字**来自对应测试文件（由 `check_ir_fidelity.py` 断言）
- [ ] 课件中出现的每个 `-sdy-*` flag 与 `passes.td` 一致（由 `check_flags.py` 断言）
- [ ] `README.md` 声明**前置课**，且前置课编号严格小于本课
- [ ] 无 JS 错误：`pageerror` 与 `console.error` 均为 0
- [ ] 无布局溢出：早/中/末三个时间点，可视区溢出元素数均为 0
- [ ] 1280×720 与 1920×1080 下均通过上一条
- [ ] 交互自检通过：播放 / 暂停 / 上一课 / 下一课 / 方向键 / 圆点跳转

### 3.2 四个门禁必须全绿

每课提交前**实际运行**并留存输出：

```sh
cd animation
python3 tools/check_coverage.py        # B 组：覆盖度
python3 tools/check_ir_fidelity.py     # A1/A3：IR 保真 + 清单一致
python3 tools/check_flags.py           # A2：flag 正确
python3 tools/check_render.py <课件路径>  # C 组：渲染质量
```

**任一门禁失败即不得提交**，不得用"下次再修"绕过。

### 3.3 内容准确性红线

- IR 片段**必须逐字复制**，禁止凭记忆改写、禁止"看起来差不多"。
- 展示时统一去除 `// CHECK` 行，只讲 `RUN` + 输入 IR + 关键输出。
- 展示 FileCheck 变量（`%[[X]]`）时必须显式标注"这是测试变量，非字面 IR"。
- **已知易错点**：不存在 `mpmd.mesh` 这个 op（mesh 只在 `#mpmd.topology` 属性与
  `!mpmd.mesh_tensor` 类型里）。凡涉及 mesh 的表述必须复核。
- **测试文件里的注释不一定是真的**：`// CHECK: x` 是 FileCheck 指令，`// CHECK x`
  （缺冒号）只是普通注释，永远不会被检查。上游测试里存在引用已删除属性的过时注释
  （L1-09 实测到一例：`type=AS`）。
  **凡"测试文件里写着但没见过实际输出"的语法，必须跑一遍 `sdy_opt` 确认**，
  不能只凭测试文件推断。
- **引用块必须与源文件保持"连续"**：测试文件常把 `// CHECK` 行插在函数签名与函数体之间。
  若把「签名 + 函数体」写进同一个 ```mlir 块，整块就**不再连续**，保真门禁会报"未逐字命中"。
  **规则**：遇到 CHECK 行夹在中间时，把签名与函数体拆成两个代码块。
  **不要手改**——直接运行 `python3 tools/split_blocks.py`（支持 `--dry-run`），
  它会自动扫描并拆分所有 `source.md`。写完 `source.md` 后先跑它，再跑保真门禁。
  **变体**：有些文件把 CHECK 行<b>逐行穿插</b>在每个算子之间（L2-07 的
  `shard_as_applies_despite_barrier`），这时每个算子行都必须**单独成块**——
  工具只能拆"签名 vs 函数体"，这种情形要手工按行拆开。
  **两种处理方式**：① 每个算子行单独成块；② 把穿插的 `// CHECK` 行**原样保留**在
  引用里（更忠实，推荐用于 `case` / `while` 这类区域算子，因为 CHECK 行正好标出了
  区域边界上的期望结果）。

---

## 4. 数量约束（保量）

- **课时数**：`TODOLIST.md` 声明的 74 课，**一课都不能少**。
- **覆盖度**：`check_coverage.py` 的差集必须为 **空**（241/241）。
- **规模断言**：某层实际测试文件数发生变化时，**必须同步更新 `TODOLIST.md` §3 矩阵**，
  并在同一次提交中说明原因。**禁止**为了让门禁变绿而删减课件覆盖。
- 允许把一课拆成多课、或多课合并，但**必须在同一次提交中同步更新 `TODOLIST.md`**，
  且 §3 矩阵总数仍为 241。
- 每完成一层，向用户报告**已完成 / 计划**的课时数与文件数对照，不得只报"进展顺利"。

---

## 5. Git 约束（每次变更都要有变更信息）

### 5.1 提交粒度

**一课一提交**（M0 单独一次提交）。禁止把多课混在一次提交里，也禁止把一课拆成多次提交。

工作区**不得长期存在未提交的课件变更**；一次会话结束时必须已提交。

### 5.2 提交信息格式（必须遵守）

```
<type>(<scope>): <简述>

覆盖: <本课覆盖的测试文件（相对 shardy/dialect/sdy 的路径，逗号分隔；超过 6 个写数量>
验收: coverage=<n>/241  fidelity=pass  flags=pass  render=<n>课件/0错误/0溢出
前置: <前置课编号，或 none>
```

- `type`：`feat`（新增课件）/ `fix`（修错）/ `refactor`（改引擎）/ `docs`（改计划/说明）/ `chore`（工具）
- `scope`：课号（如 `L1-01`）或 `shared` / `tools` / `plan`
- **`覆盖:` 与 `验收:` 两行是硬性要求**，缺一不可 —— 这是"要有 git 变更信息"的具体含义。
- 改引擎（`shared/`）时，`验收:` 行必须包含**全量课件回归**结果。

示例：

```
feat(L1-01): add mesh-and-devices lesson

覆盖: ir/test/mesh_parse_print.mlir, ir/test/mesh_verification.mlir
验收: coverage=2/241  fidelity=pass  flags=pass  render=1课件/0错误/0溢出
前置: none
```

### 5.3 禁止的 git 操作

- **禁止** `git push --force` / `--force-with-lease` 到 `main`
- **禁止** `git commit --amend` 已推送的提交
- **禁止** 修改或删除 `intro/` 的既有提交历史
- **禁止** 把 `shardy-project/` 上游克隆的内容提交进本仓库
- **禁止** 提交构建产物、截图、缓存、`.DS_Store` 等；需要时先加 `.gitignore`

---

## 6. 推送约束（必须推送到远端）

### 6.1 频率

**每完成一课（提交后）立即推送**：

```sh
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890
cd /data/WORKSPACE/shardy-project/animation
git push origin main
```

- 这是**硬要求**，不是"建议"。任何一课提交后都不得只留在本地。
- 若一次会话内完成多课，每课各自推送（禁止攒到最后一起推）。

### 6.2 推送前必做

1. 四个门禁全绿（§3.2）
2. `git status --short` 中**没有**意外的未跟踪文件
3. `git log -1 --stat` 中的文件变更与本次课件一一对应

### 6.3 推送后必做（自证）

```sh
git fetch -q origin
git rev-list --left-right --count HEAD...origin/main   # 必须输出 "0  0"
git rev-parse HEAD; git ls-remote origin main          # 两个 SHA 必须相同
```

**推送后必须实际核对远端 SHA 与本地一致**，不得只看 `git push` 的输出就宣称成功。

### 6.4 推送失败

- 先查代理与凭据（§1.2 / §1.1），不要盲目重试。
- 若确实无法推送，**必须**在给用户的报告中明确说明"本地已提交但未推送"，并给出阻塞原因；
  **禁止**隐瞒未推送状态。

---

## 7. 与上游 shardy 仓库的边界

- `/data/WORKSPACE/shardy-project/`（不含 `animation/`）是 openxla/shardy 的克隆，
  **只读引用**：可以读测试文件、可以跑 `sdy_opt` 复现结果，**不得修改任何源码**。
- 本仓库的 `.git` 在 `animation/`，与上游克隆是**两个独立仓库**，不要混淆。
- 引用上游文件时写**相对路径**（如 `shardy/dialect/sdy/ir/test/mesh_parse_print.mlir`），
  不要写机器绝对路径。

---

## 8. 禁止事项汇总

1. 禁止未通过门禁就提交。
2. 禁止提交无 `覆盖:` / `验收:` 行的 commit message。
3. 禁止提交后不推送。
4. 禁止推送后不核对远端 SHA。
5. 禁止凭记忆改写 IR。
6. 禁止修改上游 shardy 源码。
7. 禁止使用全局 git 配置（身份、凭据、代理一律仓库级）。
8. 禁止把产物/缓存/截图提交进仓库。
9. 禁止为凑覆盖率而虚报（如 `source.md` 写了某文件但课件里根本没讲）。
10. 禁止在报告中把"计划做"说成"已完成"。

---

## 9. 每课的标准执行流程

```
1. 读 TODOLIST.md 中本课的「覆盖」清单 → 逐文件读源测试文件
2. 写 source.md（逐字摘录 + 逐行注释）
3. 写 lesson.js（三段结构：看 IR / 看动画 / 做练习）
4. 写 index.html（引用 shared/）+ README.md（目标/前置/练习/验收点）
5. 跑四个门禁 → 全绿
6. git add + commit（按 §5.2 格式）
7. git push origin main
8. 核对远端 SHA（§6.3）
9. 向用户报告：课号、覆盖文件数、验收结果、commit SHA、远端 SHA
```

---

## 10. 阶段报告要求

每完成**一层**，向用户提交一份简报，必须包含：

| 项 | 要求 |
|---|---|
| 课时 | 已完成 X / 计划 Y（列出课号） |
| 文件 | 已完成覆盖 A / 本层 B（A 必须等于 B 才算完成） |
| 门禁 | `coverage` / `fidelity` / `flags` / `render` 四组的**实际输出摘要**（不是"通过"两个字） |
| git | 本层各课的 commit SHA 列表；`HEAD` 与 `origin/main` 是否一致的实测结果 |
| 偏差 | 与 `TODOLIST.md` 的任何偏离（拆课/并课/范围调整）及原因 |
| 未完成 | 明确列出还差什么，禁止模糊表述 |

**报告中的每个数字都必须来自实际命令输出。**

---

## 11. 当前进度基线（执行前请重新核对，勿直接信任）

| 项 | 状态 |
|---|---|
| `intro/` | ✅ 已完成并推送（20 幕，commit `5ec119c`） |
| `TODOLIST.md` | 📋 计划已就绪 |
| `AGENTS.md` | 📌 本文件 |
| M0 ～ M7 | ⬜ 未开始 |

> 执行前务必自行运行 `git log --oneline && git status --short && git rev-list --left-right --count HEAD...origin/main`
> 核实基线，不要假定上表仍然准确。
