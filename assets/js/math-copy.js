/* 公式复制（MathJax v3）

   排版完成后给每个 mjx-container 挂上可复制行为：
     · 点击公式 → 复制它的 LaTeX 源（行内包 $…$，行间包 $$…$$）
     · 行间公式右上角多一个「复制」按钮（hover / 键盘聚焦时出现）
     · 正文里有选区时点击不生效，避免和「选文字」「引用到评论」抢动作
     · 复制结果用页面底部的浮层反馈，1.5 秒后消失

   为什么不在渲染前抓源码：MathJax 是 async 加载的，脚本执行时 DOM 里的
   \(…\) / \[…\] 可能已经被吃掉；v3 的 MathItem 会把原始 TeX 存在 .math 上，
   排版结束（startup.promise）后读它最稳。

   两个实现细节：
     · mjx-container 的 overflow-x 是 auto，绝对定位的按钮会被裁掉，
       所以行间公式套一层 .math-copy-wrap 再定位；
     · 表单类元素才需要 tabindex，行内公式有几百个，只给行间公式加键盘焦点。
*/
(function () {
  'use strict';

  var TOAST_MS = 1500;
  var POLL_MS = 125;
  var POLL_MAX = 80;          // 100 秒上限只用于「MathJax 一直没来」的兜底

  var COPY_ICON = '<svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">' +
    '<rect x="5.6" y="5.6" width="7.9" height="7.9" rx="1.9" fill="none" stroke="currentColor" stroke-width="1.3"/>' +
    '<path d="M10.4 3.1A1.8 1.8 0 0 0 8.7 2H4.3A2.3 2.3 0 0 0 2 4.3v4.4c0 .8.5 1.5 1.2 1.7" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>' +
    '</svg>';

  /* ---------------------------------------------------------- clipboard */
  function legacyCopy(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.cssText = 'position:fixed;top:0;left:-9999px;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    done(ok);
  }

  function writeClipboard(text, done) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(
        function () { done(true); },
        function () { legacyCopy(text, done); }
      );
    } else {
      legacyCopy(text, done);
    }
  }

  /* -------------------------------------------------------------- toast */
  var toastEl = null;
  var toastTimer = 0;

  function toast(msg, bad) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.id = 'math-copy-toast';
      toastEl.setAttribute('role', 'status');
      toastEl.setAttribute('aria-live', 'polite');
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.toggle('is-bad', !!bad);
    toastEl.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toastEl.classList.remove('is-on'); }, TOAST_MS);
  }

  /* --------------------------------------------------------- 复制动作 */
  function payload(node) {
    var tex = node.getAttribute('data-tex') || '';
    return node.getAttribute('data-display') === '1' ? '$$\n' + tex + '\n$$' : '$' + tex + '$';
  }

  function doCopy(node) {
    writeClipboard(payload(node), function (ok) {
      toast(ok ? '已复制 LaTeX' : '复制失败，请手动选中', !ok);
      if (!ok) { return; }
      node.classList.add('is-copied');
      setTimeout(function () { node.classList.remove('is-copied'); }, 900);
    });
  }

  /* ------------------------------------------------------- 装饰公式节点 */
  function wrapDisplay(node) {
    var parent = node.parentNode;
    if (!parent || !parent.tagName) { return; }
    // <p> / <span> 里塞 div 会破坏结构，那种位置就只保留点击复制
    if (/^(P|SPAN|A|MJX-CONTAINER)$/.test(parent.tagName)) { return; }

    var wrap = document.createElement('div');
    wrap.className = 'math-copy-wrap';
    parent.insertBefore(wrap, node);
    wrap.appendChild(node);

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'math-copy-btn';
    btn.setAttribute('aria-label', '复制该公式的 LaTeX');
    btn.title = '复制 LaTeX';
    btn.innerHTML = COPY_ICON;
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      doCopy(node);
    });
    wrap.appendChild(btn);

    node.setAttribute('tabindex', '0');
    node.setAttribute('role', 'button');
    node.setAttribute('aria-label', '复制该公式的 LaTeX');
  }

  function decorate() {
    var doc = window.MathJax && window.MathJax.startup && window.MathJax.startup.document;
    if (!doc || !doc.math) { return 0; }
    /* MathJax 3.2 的 doc.math 是一个 MathList，真正的序列在 .list 上：
       一个**环形**双向链表（首尾是 data 为 null 的哨兵节点）。
       早期/别处的写法有 .data / .items 数组，一并兼容。 */
    var items = [];
    var list = doc.math.list;
    if (list) {
      var cursor = list.next;
      var guard = 0;
      while (cursor && cursor !== list && guard++ < 20000) {
        if (cursor.data) { items.push(cursor.data); }
        cursor = cursor.next;
      }
    } else if (doc.math.data || doc.math.items) {
      items = doc.math.data || doc.math.items;
    }
    var n = 0;

    for (var i = 0; i < items.length; i++) {
      var item = items[i];
      var node = item && item.typesetRoot;
      if (!node || node.nodeType !== 1) { continue; }
      if (!node.matches || !node.matches('mjx-container')) {
        node = node.querySelector ? node.querySelector('mjx-container') : null;
      }
      if (!node || node.hasAttribute('data-tex')) { continue; }

      var tex = item.math;
      if (!tex) { continue; }
      tex = String(tex).replace(/^\s+|\s+$/g, '');
      if (!tex) { continue; }

      var isDisplay = item.display === true || node.getAttribute('display') === 'true';
      node.setAttribute('data-tex', tex);
      node.classList.add('math-copyable');
      if (isDisplay) {
        node.setAttribute('data-display', '1');
        wrapDisplay(node);
      }
      n++;
    }
    return n;
  }

  /* --------------------------------------------------------- 事件绑定 */
  var selLen = 0;

  function bind() {
    // 记录按下瞬间的选区长度：click 触发时选区多半已经被折叠了
    document.addEventListener('mousedown', function () {
      var sel = window.getSelection ? window.getSelection() : null;
      selLen = sel ? String(sel).length : 0;
    }, true);

    document.addEventListener('click', function (e) {
      var t = e.target;
      var node = t && t.closest ? t.closest('mjx-container.math-copyable') : null;
      if (!node) { return; }
      if (node.closest('a')) { return; }     // 公式套在链接里时别抢跳转
      if (selLen > 0) { return; }            // 用户刚在选字
      e.preventDefault();
      doCopy(node);
    });

    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Enter' && e.key !== ' ') { return; }
      var node = e.target;
      if (!node || node.tagName !== 'MJX-CONTAINER') { return; }
      if (!node.hasAttribute('data-display')) { return; }
      e.preventDefault();
      doCopy(node);
    });
  }

  /* --------------------------------------------------------- 等排版结束 */
  function whenTypeset(cb) {
    var tries = 0;
    (function poll() {
      var mj = window.MathJax;
      if (mj && mj.startup && mj.startup.promise) {
        mj.startup.promise.then(cb, cb);
        return;
      }
      if (tries++ > POLL_MAX) { return; }
      setTimeout(poll, POLL_MS);
    })();
  }

  function init() {
    whenTypeset(function () {
      var n = decorate();
      // 便于排查「为什么没有按钮」：DOM 上留一个计数
      document.body.setAttribute('data-math-copy', String(n));
      if (n) { bind(); }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
