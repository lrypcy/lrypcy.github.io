/* 选中正文 → 生成引用 → 带进右侧 giscus 评论区
   --------------------------------------------------------------
   giscus 是跨源 iframe（giscus.app），父页面既读不到也写不进它的输入框，
   postMessage 只支持 setConfig（主题 / term 等）。所以这里分三步：
     1) 把选中的文字整理成 Markdown 引用（代码块内则包成 ``` 块）放进剪贴板
     2) 若右侧评论区被收起则展开（复用「评论」按钮的状态逻辑），滚动到视野
     3) 让 iframe 获得焦点并短暂高亮，用户 ⌘V / Ctrl+V 即可粘贴
   -------------------------------------------------------------- */
(function () {
  var MIN = 4;      // 少于 4 个有效字符多半是误选，不弹按钮
  var MAX = 1800;   // 超长选区截断，避免复制出一整篇文章
  var btn = null, toast = null, toastTimer = null, anchor = null;

  function isMac() {
    return /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || '');
  }

  function ensureEls() {
    if (btn) return;
    btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'pcy-quote-btn';
    btn.innerHTML =
      '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">' +
      '<path d="M7 7h4v10H7c-1.7 0-3-1.3-3-3s1.3-3 3-3V7zm10 0h4v10h-4c-1.7 0-3-1.3-3-3s1.3-3 3-3V7z"/></svg>' +
      '<span>引用评论</span>';
    // 按下时不让默认行为清掉选区
    btn.addEventListener('mousedown', function (e) { e.preventDefault(); });
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      var text = btn._text;
      hide();
      if (text) copyThenGo(text);
    });
    document.body.appendChild(btn);

    toast = document.createElement('div');
    toast.className = 'pcy-quote-toast';
    toast.setAttribute('role', 'status');
    document.body.appendChild(toast);
  }

  /* ---- 取当前选区：必须在正文内，且有一定长度 ---- */
  function currentSelection() {
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
    var body = document.querySelector('.post-body');
    if (!body) return null;
    var range = sel.getRangeAt(0);
    var node = range.commonAncestorContainer;
    if (node.nodeType !== 1) node = node.parentNode;
    if (!node || !body.contains(node)) return null;
    var text = (sel.toString() || '').replace(/\u00a0/g, ' ').trim();
    if (text.replace(/\s/g, '').length < MIN) return null;
    return { range: range, text: text, node: node };
  }

  /* ---- 生成引用文本 ---- */
  function buildQuote(sel) {
    var raw = sel.text;
    if (raw.length > MAX) raw = raw.slice(0, MAX) + '…';
    var pre = sel.node.closest ? sel.node.closest('pre') : null;
    if (pre) {
      var lang = '';
      var code = pre.querySelector('code');
      if (code && code.className) {
        var m = /language-([\w+-]+)/.exec(code.className);
        if (m && m[1] !== 'mermaid') lang = m[1];
      }
      return '```' + (lang || 'text') + '\n' + raw + '\n```\n\n';
    }
    var lines = raw.split(/\n+/).map(function (s) { return s.trim(); }).filter(Boolean);
    if (!lines.length) return '';
    return lines.map(function (s) { return '> ' + s; }).join('\n>\n') + '\n\n';
  }

  /* ---- 浮动按钮定位 ---- */
  function place(x, y) {
    var w = btn.offsetWidth, h = btn.offsetHeight;
    var left = Math.min(Math.max(8, x - w / 2), window.innerWidth - w - 8);
    var top = y - h - 10;
    if (top < 8) top = Math.min(y + 22, window.innerHeight - h - 8);
    btn.style.left = left + 'px';
    btn.style.top = top + 'px';
  }

  function show(sel, x, y) {
    ensureEls();
    btn._text = buildQuote(sel);
    if (!btn._text) { hide(); return; }
    if (typeof x === 'number') {
      anchor = { x: x, y: y };
    } else {
      var r = sel.range.getBoundingClientRect();
      anchor = { x: r.left + r.width / 2, y: r.top };
    }
    btn.classList.add('is-on');
    place(anchor.x, anchor.y);
  }

  function hide() {
    if (!btn) return;
    btn.classList.remove('is-on');
    btn._text = '';
    anchor = null;
  }

  /* ---- 复制到剪贴板（异步 API 优先，失败退回 execCommand） ---- */
  function copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-1000px';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      ta.setSelectionRange(0, text.length);
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error('copy failed'));
    });
  }

  function toastMsg(msg) {
    ensureEls();
    toast.textContent = msg;
    toast.classList.add('is-on');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.classList.remove('is-on'); }, 3600);
  }

  /* ---- 展开 + 滚动 + 聚焦评论区 ---- */
  function gotoComments() {
    var box = document.querySelector('.sidebar-comments');
    if (!box) return;
    // 评论区被收起时，走「评论」按钮的 click，状态与 localStorage 一起同步
    if (document.documentElement.classList.contains('hide-sidebar')) {
      var tg = document.getElementById('sidebarToggle');
      if (tg) tg.click();
    }
    box.scrollIntoView({ behavior: 'smooth', block: 'center' });
    box.classList.remove('quote-flash');
    void box.offsetWidth;              // 强制重排，让动画能重复触发
    box.classList.add('quote-flash');
    setTimeout(function () { box.classList.remove('quote-flash'); }, 2600);
    focusFrame(box, 0);
  }

  function focusFrame(box, tries) {
    var frame = box.querySelector('iframe.giscus-frame');
    if (!frame) {
      // giscus 是 lazy 加载，滚动过去之后 iframe 才出现，最多等 3 秒
      if (tries < 20) setTimeout(function () { focusFrame(box, tries + 1); }, 150);
      return;
    }
    try { frame.focus(); } catch (e) {}
  }

  function copyThenGo(text) {
    copy(text).then(function () {
      toastMsg('引用已复制，' + (isMac() ? '⌘V' : 'Ctrl+V') + ' 粘贴到右侧评论框');
      gotoComments();
    }, function () {
      toastMsg('复制失败，请手动复制选中的文字');
    });
  }

  /* ---- 事件 ---- */
  document.addEventListener('mouseup', function (e) {
    if (btn && btn.contains(e.target)) return;
    setTimeout(function () {
      var sel = currentSelection();
      if (!sel) { hide(); return; }
      show(sel, e.clientX, e.clientY);
    }, 0);
  });

  document.addEventListener('mousedown', function (e) {
    if (btn && !btn.contains(e.target)) hide();
  });

  // 键盘（Shift + 方向键 / Cmd+A）选择：用选区矩形定位
  document.addEventListener('keyup', function (e) {
    if (e.key === 'Escape') { hide(); return; }
    if (!e.shiftKey && e.key !== 'a' && e.key !== 'A') return;
    var sel = currentSelection();
    if (!sel) { hide(); return; }
    show(sel);
  });

  function reposition() {
    if (btn && anchor && btn.classList.contains('is-on')) place(anchor.x, anchor.y);
  }
  window.addEventListener('scroll', reposition, { passive: true });
  window.addEventListener('resize', reposition);
})();
