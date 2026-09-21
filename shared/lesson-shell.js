/* ==========================================================================
   Shardy 课件 — 外壳
   --------------------------------------------------------------------------
   用法（放在每个课件 index.html 末尾）：
     <script src="../../shared/engine.js"></script>
     <script src="../../shared/widgets.js"></script>
     <script src="../../shared/lesson-shell.js"></script>
     <script src="lesson.js"></script>
     <script>SHELL.boot(SCENES, {title: '...', kicker: '...', nav: {...}});</script>

   boot 会生成固定版式的页面骨架并启动 App，因此每课 index.html 保持极短。
   ========================================================================== */
'use strict';

const SHELL = {
  boot(scenes, opts = {}) {
    if (!document.getElementById('stage')) {
      document.body.insertAdjacentHTML('afterbegin', this.html(opts));
    }
    if (!scenes || !scenes.length) {
      document.getElementById('visual').textContent = '（本课件暂无场景）';
      return;
    }
    App.init(scenes, opts);
  },

  /* 课程导航（上一课 / 下一课），opts.nav = {prev:{href,label}, next:{href,label}} */
  navHtml(nav) {
    if (!nav || (!nav.prev && !nav.next)) return '';
    const a = (item, dir) => item
      ? `<a class="navlink" href="${item.href}" title="${item.label}">${dir === 'p' ? '◀' : ''}${item.label}${dir === 'n' ? '▶' : ''}</a>`
      : '';
    return `<div id="lesson-nav">${a(nav.prev, 'p')}${a(nav.next, 'n')}</div>`;
  },

  html(opts = {}) {
    const kicker = opts.kicker || 'Shardy 课件';
    return `
<div id="app">
  <div id="stage">
    <header id="hdr">
      <div class="row" style="justify-content:space-between;align-items:flex-start">
        <div id="scene-kicker">${kicker}</div>
        ${this.navHtml(opts.nav)}
      </div>
      <h1 id="scene-title">加载中…</h1>
      <p id="scene-sub"></p>
    </header>

    <main id="visual"></main>

    <aside id="side">
      <div id="side-cap">对应 IR</div>
      <pre id="code"></pre>
    </aside>

    <footer id="ftr">
      <div id="dots"></div>
      <div id="caption"></div>
    </footer>
  </div>

  <div id="controls">
    <button id="btn-restart" title="回到开头">⏮</button>
    <button id="btn-prev" title="上一幕 (←)">◀</button>
    <button id="btn-play" title="播放/暂停 (空格)">⏸</button>
    <button id="btn-next" title="下一幕 (→)">▶</button>
    <span id="counter">1 / 1</span>
    <div id="progress"><div id="progress-fill"></div></div>
  </div>
</div>`;
  }
};
