#!/usr/bin/env python3
"""覆盖度门禁（B 组）——断言 TODOLIST.md 规划的全部测试 IR 都被课件引用。

B1 全覆盖：上游 shardy/dialect/sdy 下所有 test/**/*.mlir，与全部课件 source.md
   中 `<!-- sdy-coverage ... -->` 声明的路径求差集 —— 必须为空。
B2 无空课：每个 source.md 至少声明 1 个文件。
B3 无幻影：声明的路径必须真实存在。

用法：
  python3 tools/check_coverage.py
  python3 tools/check_coverage.py --upstream ../shardy/dialect/sdy
  python3 tools/check_coverage.py --quiet
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UPSTREAM = os.path.normpath(os.path.join(ROOT, "..", "shardy", "dialect", "sdy"))

# source.md 中的机器可读声明块
COV_RE = re.compile(r"<!--\s*sdy-coverage\s*(.*?)-->", re.S)


def upstream_files(root):
    """上游 test 目录下的全部 .mlir（相对 root 的路径）。"""
    out = set()
    for dp, _, fs in os.walk(root):
        parts = dp.split(os.sep)
        if "test" not in parts:
            continue
        for f in fs:
            if f.endswith(".mlir"):
                out.add(os.path.relpath(os.path.join(dp, f), root))
    return out


def lesson_source_files():
    """返回 {source.md 相对路径: set(声明路径)}。"""
    out = {}
    for dp, dns, fs in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "shared", "tools", "node_modules")]
        if "source.md" not in fs:
            continue
        p = os.path.join(dp, "source.md")
        txt = open(p, encoding="utf-8").read()
        declared = set()
        for m in COV_RE.finditer(txt):
            for line in m.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    declared.add(line.lstrip("./"))
        out[os.path.relpath(p, ROOT)] = declared
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.upstream):
        print("✗ 找不到上游目录：%s" % a.upstream)
        return 2

    universe = upstream_files(a.upstream)
    lessons = lesson_source_files()

    covered, empty_lessons, phantom = set(), [], []
    for rel, declared in sorted(lessons.items()):
        if not declared:
            empty_lessons.append(rel)
        for d in declared:
            if d in universe:
                covered.add(d)
            else:
                phantom.append((rel, d))

    missing = sorted(universe - covered)
    total, ncov = len(universe), len(covered)
    pct = (100.0 * ncov / total) if total else 0.0

    print("上游测试 IR 总数 : %d" % total)
    print("已被课件覆盖     : %d  (%.1f%%)" % (ncov, pct))
    print("课件 source.md 数: %d" % len(lessons))

    if not a.quiet:
        if missing:
            print("\n未覆盖文件（%d）：" % len(missing))
            for m in missing[:60]:
                print("  - " + m)
            if len(missing) > 60:
                print("  … 还有 %d 个" % (len(missing) - 60))

    ok = True
    if missing:
        print("\n✗ B1 覆盖度不足：%d 个文件没有任何课件引用" % len(missing))
        ok = False
    if empty_lessons:
        print("\n✗ B2 空课件（source.md 未声明任何文件）：")
        for e in empty_lessons:
            print("  - " + e)
        ok = False
    if phantom:
        print("\n✗ B3 引用了不存在的文件：")
        for rel, d in phantom[:20]:
            print("  - %s  ->  %s" % (rel, d))
        ok = False

    if ok:
        print("\n✓ B 组通过：%d/%d 全覆盖，无空课，无幻影引用" % (ncov, total))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
