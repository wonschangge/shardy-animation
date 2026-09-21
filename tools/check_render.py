#!/usr/bin/env python3
"""渲染门禁（C 组）——逐课件、逐幕验证渲染质量。

断言：
  C1 无 pageerror / console.error
  C2 无布局溢出（早/中/末三个时间点，可视区元素不得越界）
  C4 交互可用（播放/暂停/上一幕/下一幕/方向键/圆点）
  C5 资源无 404

用法：
  python3 tools/check_render.py --all
  python3 tools/check_render.py intro L1-ir/L1-01-mesh
  python3 tools/check_render.py --all --res 1280x720 1920x1080
"""
import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/usr/bin/google-chrome"

OVERFLOW_JS = """(()=>{
  const v = document.getElementById('visual').getBoundingClientRect();
  const bad = [];
  document.querySelectorAll('#visual *').forEach(el=>{
    const b = el.getBoundingClientRect();
    if (b.width === 0 && b.height === 0) return;
    const over = Math.round(Math.max(v.top-b.top, b.bottom-v.bottom, v.left-b.left, b.right-v.right));
    if (over > 3) bad.push((el.className.toString().slice(0,30) || el.tagName) + ':' + over);
  });
  return {n: bad.length, sample: bad.slice(0,3)};
})()"""

STATE_JS = """(()=>{
  const vis = document.getElementById('visual');
  return {
    n: (typeof SCENES !== 'undefined') ? SCENES.length : 0,
    buildErr: vis.innerText.startsWith('场景渲染出错'),
    nodes: vis.querySelectorAll('*').length,
    title: (document.getElementById('scene-title')||{}).innerText || '',
    codeLen: (document.getElementById('code')||{innerText:''}).innerText.length
  };
})()"""


def discover():
    """返回所有含 index.html 的课件目录（相对 ROOT）。"""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "*", "*", "index.html"))):
        out.append(os.path.relpath(os.path.dirname(p), ROOT))
    if os.path.exists(os.path.join(ROOT, "intro", "index.html")):
        out.insert(0, "intro")
    return out


def check_one(page, rel, res, tick_points, strict_interaction):
    """返回 (ok, list_of_problems, stats)"""
    url = "file://" + os.path.join(ROOT, rel, "index.html")
    problems, console_errors, notfound = [], [], []

    def on_console(m):
        if m.type == "error":
            console_errors.append(m.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: console_errors.append("PAGEERROR: " + str(e)))
    page.on("requestfailed", lambda r: notfound.append(r.url))
    page.on("response", lambda r: notfound.append("%s %d" % (r.url, r.status)) if r.status >= 400 else None)

    page.set_viewport_size(res)
    page.goto(url, wait_until="load")
    page.wait_for_timeout(500)

    st = page.evaluate(STATE_JS)
    if st["n"] == 0:
        problems.append("SCENES 为空或未加载")
        return False, problems, st

    worst = 0
    for i in range(st["n"]):
        for ms in tick_points:
            page.evaluate("App.go(%d); App.playing=false; App.tl.tick(%d);" % (i, ms))
            page.wait_for_timeout(120)
            s = page.evaluate(STATE_JS)
            if s["buildErr"]:
                problems.append("第 %d 幕构建报错" % (i + 1))
            if s["nodes"] < 3:
                problems.append("第 %d 幕可视区为空 (%d 节点)" % (i + 1, s["nodes"]))
            o = page.evaluate(OVERFLOW_JS)
            if o["n"]:
                worst = max(worst, o["n"])
                problems.append("第 %d 幕 @%dms 溢出 %d 处 %s" % (i + 1, ms, o["n"], o["sample"]))

    # 交互自检
    if strict_interaction:
        page.evaluate("App.go(0); App.playing=true;")
        page.wait_for_timeout(150)
        page.click("#btn-next"); page.wait_for_timeout(120)
        if page.evaluate("App.idx") != 1:
            problems.append("下一幕按钮无效")
        page.click("#btn-prev"); page.wait_for_timeout(120)
        if page.evaluate("App.idx") != 0:
            problems.append("上一幕按钮无效")
        page.click("#btn-play"); page.wait_for_timeout(120)
        if page.evaluate("App.playing") is not False:
            problems.append("播放/暂停切换无效")
        page.click("#btn-play"); page.wait_for_timeout(120)
        page.keyboard.press("ArrowRight"); page.wait_for_timeout(120)
        if page.evaluate("App.idx") != 1:
            problems.append("方向键翻页无效")
        page.click("#btn-restart"); page.wait_for_timeout(120)
        if page.evaluate("App.idx") != 0:
            problems.append("回到开头无效")
        page.evaluate("App.playing=false;")
        # 圆点跳转
        n = page.evaluate("document.querySelectorAll('.dot').length")
        if n != st["n"]:
            problems.append("圆点数 %d != 幕数 %d" % (n, st["n"]))

    if console_errors:
        problems.append("控制台错误 %d 条：%s" % (len(console_errors), console_errors[:2]))
    if notfound:
        problems.append("资源加载失败：%s" % notfound[:3])

    st["overflow_scenes"] = worst
    return (not problems), problems, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", help="课件目录（相对 animation/）")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--res", nargs="+", default=["1440x900"])
    ap.add_argument("--ticks", nargs="+", type=int, default=[1500, 6000, 20000])
    ap.add_argument("--no-interaction", action="store_true")
    a = ap.parse_args()

    targets = discover() if a.all else a.lessons
    if not targets:
        print("没有指定课件（用 --all 或传入目录）")
        return 2

    resolutions = []
    for r in a.res:
        w, h = r.lower().split("x")
        resolutions.append({"width": int(w), "height": int(h)})

    from playwright.sync_api import sync_playwright
    fails = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME)
        for rel in targets:
            if not os.path.exists(os.path.join(ROOT, rel, "index.html")):
                print("SKIP  %-34s 无 index.html" % rel)
                continue
            for res in resolutions:
                page = browser.new_page(viewport=res)
                try:
                    ok, problems, st = check_one(page, rel, res, a.ticks, not a.no_interaction)
                finally:
                    page.close()
                tag = "%dx%d" % (res["width"], res["height"])
                if ok:
                    print("PASS  %-34s %-10s scenes=%-3d code=%s" %
                          (rel, tag, st["n"], "yes" if st.get("codeLen") else "no"))
                else:
                    fails += 1
                    print("FAIL  %-34s %-10s" % (rel, tag))
                    for pr in problems[:8]:
                        print("        - " + pr)
        browser.close()

    print("\n%s  课件 %d 个，失败 %d" % ("✓ 全部通过" if fails == 0 else "✗ 存在失败", len(targets), fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
