#!/usr/bin/env python3
"""flag 正确性门禁（A2）——断言课件中出现的每个 pass flag 都真实存在。

真值来源（按优先级）：
  1. 已构建的 sdy_opt / mpmd_opt 的 `--help` 输出（最权威）
  2. 回退：解析 shardy 的 */passes.td 与 docs/*_passes.md

检查对象：animation/ 下所有课件与文档（排除 shared/、tools/、.git/）。

用法：
  python3 tools/check_flags.py
  python3 tools/check_flags.py --sdy-opt /path/to/sdy_opt
"""
import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.normpath(os.path.join(ROOT, ".."))
DEFAULT_SDY = os.path.join(UPSTREAM, "bazel-bin", "shardy", "tools", "sdy_opt")
DEFAULT_MPMD = os.path.join(UPSTREAM, "bazel-bin", "shardy", "tools", "mpmd_opt")

PASS_RE = re.compile(r"(?<![\w-])-(sdy|mpmd)-([a-z0-9][a-z0-9-]*)")
# -sdy-xxx=opt=val opt2=val2  -> 取 opt / opt2
OPT_RE = re.compile(r"-(?:sdy|mpmd)-[a-z0-9-]+=([^\s`\"'<>]+)")


def help_text(binary):
    if not (binary and os.path.exists(binary)):
        return None
    try:
        r = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr
    except Exception:
        return None


def truth_from_help():
    passes, options = set(), set()
    got = False
    for b, ns in ((DEFAULT_SDY, "sdy"), (DEFAULT_MPMD, "mpmd")):
        txt = help_text(b)
        if txt is None:
            continue
        got = True
        for m in re.finditer(r"^\s+--([a-z0-9][a-z0-9-]*)", txt, re.M):
            name = m.group(1)
            if name.startswith(ns + "-"):
                passes.add(name)
            options.add(name)
    return (passes, options) if got else (None, None)


def truth_from_source():
    """回退：从 passes.td / docs 抽取。"""
    passes, options = set(), set()
    for pat in ("shardy/dialect/*/transforms/*/passes.td",
                "shardy/dialect/*/*/passes.td",
                "docs/*_passes.md",
                "docs/*/*_passes.md"):
        for f in glob.glob(os.path.join(UPSTREAM, pat)):
            txt = open(f, encoding="utf-8", errors="replace").read()
            if f.endswith(".md"):
                for m in re.finditer(r"^###\s+`(-[a-z0-9-]+)`", txt, re.M):
                    passes.add(m.group(1).lstrip("-"))
                for m in re.finditer(r"^-\s+`([a-z0-9-]+)`", txt, re.M):
                    options.add(m.group(1))
            else:
                for m in re.finditer(r'let\s+option\s*=\s*"([a-z0-9-]+)"', txt):
                    options.add(m.group(1))
                for m in re.finditer(r'def\s+\w+\s*:\s*Pass<"([^"]+)"', txt):
                    options.add(m.group(1).split("-")[-1])
    return passes, options


def scan_files():
    out = []
    for dp, dns, fs in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "shared", "tools", "node_modules")]
        for f in fs:
            if f.endswith((".js", ".md", ".html", ".css")):
                out.append(os.path.join(dp, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdy-opt", default=DEFAULT_SDY)
    ap.add_argument("--mpmd-opt", default=DEFAULT_MPMD)
    ap.add_argument("--max-report", type=int, default=15)
    ap.add_argument("--quiet", action="store_true", help="只输出结论")
    a = ap.parse_args()

    passes, options = truth_from_help()
    src = "sdy_opt/mpmd_opt --help"
    if passes is None:
        passes, options = truth_from_source()
        src = "passes.td / docs（回退）"
    if not passes and not options:
        print("✗ 无法获得 flag 真值（既没有构建产物，也没有源码）")
        return 2
    if not a.quiet:
        print("flag 真值来源：%s（pass %d 个，选项 %d 个）" % (src, len(passes), len(options)))

    bad_pass, bad_opt, n_pass, n_opt = [], [], 0, 0
    for f in scan_files():
        txt = open(f, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(f, ROOT)
        for m in PASS_RE.finditer(txt):
            name = "%s-%s" % (m.group(1), m.group(2))
            n_pass += 1
            if name not in passes:
                bad_pass.append((rel, name))
        for m in OPT_RE.finditer(txt):
            for part in m.group(1).split():
                if "=" not in part:
                    continue
                opt = part.split("=", 1)[0]
                if not opt or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", opt):
                    continue
                n_opt += 1
                if opt not in options and opt not in passes:
                    bad_opt.append((rel, opt))

    if not a.quiet:
        print("扫描到 pass flag 引用 %d 处，选项引用 %d 处" % (n_pass, n_opt))

    if bad_pass or bad_opt:
        print("\n✗ A2 失败：")
        seen = set()
        for rel, name in bad_pass[:a.max_report]:
            if (rel, name) in seen:
                continue
            seen.add((rel, name))
            print("  - [%s] 不存在的 pass flag: -%s" % (rel, name))
        for rel, opt in bad_opt[:a.max_report]:
            if (rel, opt) in seen:
                continue
            seen.add((rel, opt))
            print("  - [%s] 不存在的 pass 选项: %s" % (rel, opt))
        return 1

    print("\n✓ A2 通过：全部 flag 与选项均真实存在")
    return 0


if __name__ == "__main__":
    sys.exit(main())
