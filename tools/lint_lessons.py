#!/usr/bin/env python3
"""对所有课件做一次静态语法检查（比渲染门禁更快、更早发现问题）。

为什么需要它
------------
写 `lesson.js` 时容易犯一类低级但致命的笔误：

    U.el('div', { style: 'gap':13px;width:100%' })
                            ^^^^^ 引号位置错 -> SyntaxError

这类错误会让整个课件白屏。渲染门禁（check_render.py）能发现，但要启动
浏览器、慢得多。本脚本用 `node --check` 做纯静态检查，秒级返回。

用法：
  python3 tools/lint_lessons.py            # 检查全部 lesson.js
  python3 tools/lint_lessons.py --quiet    # 只输出结论
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 已知的高频笔误模式（在 node --check 之外额外提示，便于快速定位）
SUSPECT = [
    (re.compile(r"style:\s*'[a-z-]+'\s*:"), "style 字符串引号位置错误（应为 style: 'a:b'）"),
    (re.compile(r"style:\s*'[^']*'\s*[a-z-]+\s*:"), "style 字符串提前闭合"),
]


def find_lessons():
    out = []
    for dp, dns, fs in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "shared", "tools", "node_modules")]
        if "lesson.js" in fs:
            out.append(os.path.join(dp, "lesson.js"))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    lessons = find_lessons()
    bad = 0
    for p in lessons:
        rel = os.path.relpath(p, ROOT)
        r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
        if r.returncode != 0:
            bad += 1
            print("✗ %s" % rel)
            msg = (r.stderr or "").strip().split("\n")
            for line in msg[:4]:
                print("    %s" % line)
            continue
        # 语法通过，再查可疑模式（给出更友好的提示）
        src = open(p, encoding="utf-8").read()
        for rx, desc in SUSPECT:
            for m in rx.finditer(src):
                ln = src[:m.start()].count("\n") + 1
                print("⚠ %s:%d  %s" % (rel, ln, desc))
                break

    if bad:
        print("\n✗ %d/%d 个课件存在语法错误" % (bad, len(lessons)))
        return 1
    if not a.quiet:
        print("✓ 全部 %d 个 lesson.js 语法通过" % len(lessons))
    else:
        print("✓ lint ok (%d)" % len(lessons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
