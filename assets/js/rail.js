/* 左侧目录（rail）：展开状态记忆、当前文章定位、整体收起 / 展开 */
(function () {
  var KEY = 'pcy-rail-pref';
  var rail = document.getElementById('rail');
  if (!rail) { return; }

  var pref = { open: [], collapsed: false };
  try {
    var saved = JSON.parse(localStorage.getItem(KEY) || '{}');
    if (saved && Array.isArray(saved.open)) { pref.open = saved.open; }
    if (saved && typeof saved.collapsed === 'boolean') { pref.collapsed = saved.collapsed; }
  } catch (e) {}
  function save() { try { localStorage.setItem(KEY, JSON.stringify(pref)); } catch (e) {} }

  /* 1. 分类展开 / 收起：记忆在 localStorage，当前文章所属分类始终展开 */
  var cats = [].slice.call(rail.querySelectorAll('.rail-cat'));
  cats.forEach(function (d) {
    var name = d.getAttribute('data-cat');
    var isCurrent = d.getAttribute('data-current') === '1';
    d.open = isCurrent || pref.open.indexOf(name) >= 0;
    d.addEventListener('toggle', function () {
      var i = pref.open.indexOf(name);
      if (d.open && i < 0) { pref.open.push(name); }
      else if (!d.open && i >= 0) { pref.open.splice(i, 1); }
      save();
    });
  });

  /* 2. 当前文章：滚到目录可视范围内（只动目录容器的 scrollTop，不动页面） */
  var current = rail.querySelector('.rail-posts a.is-current');
  var scroll = document.getElementById('railScroll');
  if (current && scroll) {
    var jump = function () {
      var top = current.offsetTop - scroll.clientHeight / 2;
      scroll.scrollTop = top > 0 ? top : 0;
    };
    jump();
    if (window.requestAnimationFrame) { window.requestAnimationFrame(jump); }
  }

  /* 3. 整体收起 / 展开 */
  var btn = document.getElementById('railCollapse');
  function applyCollapsed() {
    document.documentElement.classList.toggle('rail-off', pref.collapsed);
    if (!btn) { return; }
    var icon = btn.querySelector('.rail-toggle-icon');
    var text = btn.querySelector('.rail-toggle-text');
    if (icon) { icon.textContent = pref.collapsed ? '»' : '«'; }
    if (text) { text.textContent = pref.collapsed ? '展开' : '收起'; }
    btn.title = pref.collapsed ? '展开目录' : '收起目录';
    btn.setAttribute('aria-expanded', pref.collapsed ? 'false' : 'true');
    // 目录宽度变了，让文章页的悬浮章节线重新定位（见 _layouts/post.html）
    if (window.PCYToc && window.PCYToc.reposition) {
      if (window.requestAnimationFrame) { window.requestAnimationFrame(window.PCYToc.reposition); }
      else { window.PCYToc.reposition(); }
    }
  }
  if (btn) {
    btn.addEventListener('click', function () {
      pref.collapsed = !pref.collapsed;
      applyCollapsed();
      save();
    });
  }
  applyCollapsed();
})();
