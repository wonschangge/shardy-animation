#!/usr/bin/env python3
"""把 source.md 中「函数签名 + 函数体」的 ```mlir 块拆成两个块。

为什么需要它
------------
测试文件常把 `// CHECK` 行插在函数签名与函数体之间，例如：

    func.func @foo(...) {          <- 签名
      // CHECK-NEXT: ...           <- 期望输出（夹在中间）
      %0 = stablehlo.add ...       <- 函数体
      return %0 : ...
    }

若把「签名 + 函数体」写进同一个 ```mlir 块，该块在源文件里就**不再是连续片段**，
`check_ir_fidelity.py` 会报"未逐字命中"。本脚本自动把它们拆开。

用法：
  python3 tools/split_blocks.py            # 处理所有 source.md
  python3 tools/split_blocks.py --dry-run  # 只报告，不修改
  python3 tools/split_blocks.py <path>...  # 只处理指定文件
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = re.compile(r"```mlir\n(.*?)```", re.S)


def split_block(m):
    """签名（以 '{' 结尾，可能跨多行）与之后的函数体拆成两块。"""
    lines = m.group(1).rstrip("\n").split("\n")
    if not lines:
        return m.group(0), False
    # 允许块以注释 / CHECK 行开头：找到第一条 func.func 所在位置
    start = 0
    while start < len(lines) and not lines[start].lstrip().startswith("func.func"):
        if lines[start].strip() and not lines[start].lstrip().startswith("//"):
            return m.group(0), False   # 首个非注释行不是 func.func -> 不处理
        start += 1
    if start >= len(lines):
        return m.group(0), False
    # 找到签名结束行（以 '{' 结尾）
    i = start
    while i < len(lines) and not lines[i].rstrip().endswith("{"):
        i += 1
    if i >= len(lines) - 1:          # 没有函数体，无需拆分
        return m.group(0), False
    sig, body = lines[:i + 1], lines[i + 1:]
    if not body or not any(l.strip() for l in body):
        return m.group(0), False
    return ("```mlir\n" + "\n".join(sig) + "\n```\n\n```mlir\n" +
            "\n".join(body) + "\n```"), True


def process(path, dry):
    src = open(path, encoding="utf-8").read()
    n = [0]

    def repl(m):
        out, changed = split_block(m)
        if changed:
            n[0] += 1
        return out

    new = BLOCK.sub(repl, src)
    if n[0] and not dry:
        open(path, "w", encoding="utf-8").write(new)
    return n[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    targets = a.paths
    if not targets:
        targets = []
        for dp, dns, fs in os.walk(ROOT):
            dns[:] = [d for d in dns if d not in (".git", "shared", "tools", "node_modules")]
            if "source.md" in fs:
                targets.append(os.path.join(dp, "source.md"))
    targets.sort()

    total, touched = 0, 0
    for p in targets:
        if not os.path.exists(p):
            print("跳过（不存在）：%s" % p)
            continue
        k = process(p, a.dry_run)
        if k:
            touched += 1
            total += k
            print("%s  %s %d 个块" % ("[dry-run] 需拆分" if a.dry_run else "已拆分", os.path.relpath(p, ROOT), k))

    if not total:
        print("✓ 无需拆分：所有 source.md 的 func 块都已与函数体分离")
    else:
        print("\n%s：%d 个文件 / %d 个块" % ("待拆分" if a.dry_run else "已处理", touched, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
