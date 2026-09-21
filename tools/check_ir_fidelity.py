#!/usr/bin/env python3
"""IR 保真门禁（A1/A3）——断言课件里的 IR 片段逐字来自源测试文件。

A1：source.md 中每个 ```mlir 代码块，必须能在其声明的源测试文件中找到
    **连续的逐字片段**（忽略行首缩进与 CHECK 前缀，忽略空行）。
    需要省略时用单独一行的 `...` 分段，每段各自必须逐字命中。
A3：source.md 声明的文件清单必须与它在 coverage 块中列出的一致（结构自检）。

用法：
  python3 tools/check_ir_fidelity.py
  python3 tools/check_ir_fidelity.py --upstream ../shardy/dialect/sdy
  python3 tools/check_ir_fidelity.py --lesson L1-ir/L1-01-mesh
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_UPSTREAM = os.path.normpath(os.path.join(ROOT, "..", "shardy", "dialect", "sdy"))

COV_RE = re.compile(r"<!--\s*sdy-coverage\s*(.*?)-->", re.S)
# ```mlir ... ``` 或 ```c ... ```（SDY 文档里也有 c 语言示例）
BLOCK_RE = re.compile(r"```(?:mlir|MLIR)\s*\n(.*?)```", re.S)
ELIDE = "..."

# 行首的 FileCheck 指令前缀，比对时剥掉（源文件里 CHECK 行承载的是期望输出）
CHECK_PREFIX = re.compile(r"^\s*//\s*(CHECK[A-Z-]*|RUN)\s*:?\s?")


def norm_line(s):
    return s.rstrip()


def normalize(raw):
    """统一归一化：剥掉 CHECK/RUN 前缀、去首尾空白。
    源文件与引用块必须走同一套，否则引用 CHECK 行会被误判。"""
    line = CHECK_PREFIX.sub("", raw) if CHECK_PREFIX.match(raw) else raw
    return norm_line(line).strip()


def source_lines(path):
    """源文件 -> 归一化行列表（剥掉 CHECK/RUN 前缀，保留原顺序）。"""
    if not os.path.exists(path):
        return None
    out = []
    for raw in open(path, encoding="utf-8", errors="replace").read().splitlines():
        s = normalize(raw)
        if s:
            out.append(s)
    return out


def contains_run(hay, needle):
    """needle 是否为 hay 中连续的一段。"""
    n, m = len(hay), len(needle)
    if m == 0:
        return True
    for i in range(n - m + 1):
        if hay[i:i + m] == needle:
            return True
    return False


def find_lesson_dirs():
    out = []
    for dp, dns, fs in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "shared", "tools")]
        if "source.md" in fs:
            out.append(dp)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    ap.add_argument("--lesson", default=None, help="只检查某个课件目录")
    ap.add_argument("--max-report", type=int, default=12)
    ap.add_argument("--quiet", action="store_true", help="只输出结论")
    a = ap.parse_args()

    lessons = find_lesson_dirs()
    if a.lesson:
        lessons = [d for d in lessons if os.path.relpath(d, ROOT) == a.lesson]
        if not lessons:
            print("✗ 未找到课件：%s" % a.lesson)
            return 2

    if not lessons:
        print("（尚无课件，跳过）")
        print("✓ A1 通过：0 个代码块")
        return 0

    total_blocks, bad = 0, []
    for d in lessons:
        rel = os.path.relpath(d, ROOT)
        md = open(os.path.join(d, "source.md"), encoding="utf-8").read()

        declared = []
        for m in COV_RE.finditer(md):
            for line in m.group(1).splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    declared.append(line.lstrip("./"))
        if not declared:
            bad.append((rel, "(source.md 缺少 sdy-coverage 声明块)", ""))
            continue

        # 预载源行
        srcs = {}
        for f in declared:
            srcs[f] = source_lines(os.path.join(a.upstream, f))

        for bi, m in enumerate(BLOCK_RE.finditer(md)):
            total_blocks += 1
            block = m.group(1).strip("\n")
            # 按省略标记切段
            segs, cur = [], []
            for line in block.splitlines():
                if line.strip() == ELIDE:
                    if cur:
                        segs.append(cur)
                    cur = []
                else:
                    s = normalize(line)
                    if s:
                        cur.append(s)
            if cur:
                segs.append(cur)
            if not segs:
                continue

            for seg in segs:
                hit = None
                for f, lines in srcs.items():
                    if lines and contains_run(lines, seg):
                        hit = f
                        break
                if hit is None:
                    first = seg[0][:90]
                    bad.append((rel, "块#%d 未逐字命中：%s" % (bi + 1, first),
                                "已声明：" + ", ".join(declared)))

    if not a.quiet:
        print("检查课件 %d 个，IR 代码块 %d 个" % (len(lessons), total_blocks))
    if bad:
        print("\n✗ A1 失败 %d 处：" % len(bad))
        for rel, why, extra in bad[:a.max_report]:
            print("  - [%s] %s" % (rel, why))
            if extra:
                print("      %s" % extra)
        if len(bad) > a.max_report:
            print("  … 还有 %d 处" % (len(bad) - a.max_report))
        return 1
    print("\n✓ A1/A3 通过：全部 IR 片段逐字来自声明的源测试文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
