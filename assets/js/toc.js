/* 章节目录（文章页 _layouts/post.html 与资讯页 _layouts/daily.html 共用）

   pcyToc({
     selector : 'h2',      收进目录的标题（在 .post-body 内查），可写 'h2, h3'
     minItems : 3,         少于这个**条目数**就不建目录（分组模式下不含分类节）
     grouped  : false,     true = h2 当分组标题、其余当条目（资讯页：分类节 + 条目）
     float    : true       false = 永远不用左侧悬浮章节线，只用卡片式目录
     minRoom  : 0          >0 时，正文左侧留白不足就不摆悬浮目录，退回卡片
   })

   三段式呈现，与主题既有行为一致：
     宽屏 (>=992)  左侧竖排章节线，hover 原位展开成文字，滚动高亮当前节
     中屏          右侧 sidebar 顶部目录卡片
     窄屏 (<769)   正文顶部可折叠目录卡
   断点必须与 custom.css 里 `.side-toc { display:none }` / 布局断点保持一致，
   否则会出现「JS 建了悬浮目录、CSS 又把它藏掉」的空目录。

   rail.js 收起/展开左栏、post.html 切换评论区时都回调 window.PCYToc.reposition。
*/
(function () {
  var BREAK_FLOAT = 992;
  var BREAK_SIDE = 769;

  window.pcyToc = function (opts) {
    opts = opts || {};
    var body = document.querySelector('.post-body');
    if (!body) { return; }

    var sel = opts.selector || 'h2';
    var min = typeof opts.minItems === 'number' ? opts.minItems : 3;
    var grouped = !!opts.grouped;
    var minRoom = opts.minRoom || 0;
    var allowFloat = opts.float !== false;

    var hs = [].slice.call(body.querySelectorAll(sel)).filter(function (h) {
      // 目录卡自己也在 .post-body 里，别把自己收进目录
      return !h.closest('.post-toc') && !h.closest('.sidebar-toc');
    });
    // minItems 数的是**条目**：分组模式下分类节标题不该把「1 类 2 条」的短页凑成门槛
    var itemCount = grouped ? hs.filter(function (h) { return h.tagName !== 'H2'; }).length : hs.length;
    if (itemCount < min) { return; }

    // 锚点：正文已有的 id 一律沿用（资讯页分类节是流水线写的 cat-<key>），缺的才补。
    hs.forEach(function (h) {
      if (!h.id) { h.id = 'sec-' + Math.random().toString(36).slice(2, 8); }
    });

    /* 资讯页按所属分类上色：色值来自 custom.css 的 [data-cat] 调色板。
       条目的分类 = 它上面最近的那个 h2。 */
    function catOf(el) {
      if (!grouped) { return ''; }
      // 分节标题只认自己的 id：往回找会命中**上一个**分类节，
      // 于是第 2 个以后的分节都被涂上前一类的颜色。
      if (el.tagName === 'H2') {
        var own = el.id || '';
        return own.indexOf('cat-') === 0 ? own.slice(4) : '';
      }
      var cur = el.previousElementSibling;
      while (cur) {
        if (cur.tagName === 'H2') {
          var id = cur.id || '';
          return id.indexOf('cat-') === 0 ? id.slice(4) : '';
        }
        cur = cur.previousElementSibling;
      }
      return '';
    }

    function isSection(el) { return grouped && el.tagName === 'H2'; }

    var view = null;   // { kind:'float'|'card', destroy() }

    /* 摆放位置：
       1. 有左侧 rail（文章页）→ 贴到 rail 右缘的缝隙，与改造前逐字一致；
       2. 没有 rail（资讯页刻意不预留 rail 列）→ 落进正文卡片自己的左内边距里，
          收起态只有 28px 宽（padding 6 + 横线 16），卡片内边距 40px 装得下，
          既贴边又不压到文字；hover 展开时盖住正文，与文章页悬浮目录同套路。
          内边距不够才退回卡片外侧。 */
    function hasRoom() {
      if (!minRoom) { return true; }
      var rail = document.getElementById('rail');
      if (rail && rail.offsetWidth > 0) { return true; }
      return cardPaddingRoom() >= 34 || outsideRoom();
    }

    function cardPaddingRoom() {
      var block = body.closest('.post-block') || body;
      return parseFloat(getComputedStyle(block).paddingLeft || '0') || 0;
    }

    function outsideRoom() {
      var main = document.querySelector('.layout-main') || body;
      return main.getBoundingClientRect().left >= minRoom;
    }

    function wantFloating() {
      return allowFloat && window.innerWidth >= BREAK_FLOAT && hasRoom();
    }

    function smoothTo(target) {
      if (target) { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    }

    function activeIndex() {
      var y = window.scrollY + 110, idx = -1;
      for (var i = 0; i < hs.length; i++) {
        if (hs[i].getBoundingClientRect().top + window.scrollY <= y) { idx = i; } else { break; }
      }
      return idx;
    }

    function spy(marks) {
      var idx = activeIndex();
      marks.forEach(function (el, j) { el.classList.toggle('active', j === idx); });
    }

    function buildFloating() {
      var wrap = document.createElement('div');
      wrap.className = 'side-toc';
      var lines = document.createElement('div');
      lines.className = 'side-toc-lines';
      var items = [], n = 0;

      hs.forEach(function (h, i) {
        var sec = isSection(h);
        var item = document.createElement('div');
        item.className = 'side-toc-item' + (sec ? ' is-section' : '');
        item.setAttribute('data-idx', i);
        item.title = h.textContent;
        var key = catOf(h);
        if (key) { item.setAttribute('data-cat', key); }

        var bar = document.createElement('span');
        bar.className = 'side-toc-bar';
        var label = document.createElement('span');
        label.className = 'side-toc-label';
        if (sec) {
          label.textContent = h.textContent;
        } else {
          n += 1;
          label.textContent = String(n).padStart(2, '0') + '  ' + h.textContent;
        }
        item.appendChild(bar);
        item.appendChild(label);
        lines.appendChild(item);
        items.push(item);
      });
      wrap.appendChild(lines);
      document.body.appendChild(wrap);

      function place() {
        var rail = document.getElementById('rail');
        if (rail && rail.offsetWidth > 0) {
          // 文章页：贴到左栏右缘的缝隙里（与既有行为一致）
          wrap.classList.add('side-toc-shifted');
          wrap.style.left = Math.round(rail.getBoundingClientRect().right + 4) + 'px';
          return;
        }
        wrap.classList.remove('side-toc-shifted');
        if (!minRoom) { wrap.style.left = ''; return; }   // 文章页：沿用 CSS 默认落点，行为与改造前一致
        var block = body.closest('.post-block') || body;
        var bRect = block.getBoundingClientRect();
        var padL = parseFloat(getComputedStyle(block).paddingLeft || '0') || 0;
        if (padL >= 34) {
          /* 落在卡片左内边距里。留白计算要按**最宽的那一档**：横线默认 16px、
             active 态 26px，外加 .side-toc-lines 的 6px 左内边距，
             所以 2 + 6 + 26 = 34 才是收起态的真实右缘。 */
          wrap.style.left = Math.round(bRect.left + 2) + 'px';
        } else {
          var main = document.querySelector('.layout-main') || body;
          wrap.style.left = Math.max(12, Math.round(main.getBoundingClientRect().left) - 46) + 'px';
        }
      }
      place();

      function onClick(e) {
        var it = e.target.closest('.side-toc-item');
        if (!it) { return; }
        smoothTo(hs[+it.getAttribute('data-idx')]);
      }
      function onScroll() { spy(items); }
      lines.addEventListener('click', onClick);
      window.addEventListener('scroll', onScroll, { passive: true });
      setTimeout(onScroll, 100);

      return {
        kind: 'float',
        reposition: place,
        destroy: function () {
          window.removeEventListener('scroll', onScroll);
          if (wrap.parentNode) { wrap.parentNode.removeChild(wrap); }
        }
      };
    }

    function buildCard(inline) {
      var toc = document.createElement(inline ? 'details' : 'div');
      toc.className = inline ? 'post-toc' : 'sidebar-toc';
      if (inline) { toc.open = false; }   // 手机端默认折叠（只在创建时设，重新安置不打断阅读）
      var title = document.createElement(inline ? 'summary' : 'div');
      title.textContent = inline ? '📑 目录 · ' + itemCount + ' 节' : '📑 目录';
      var ul = document.createElement('ul');
      var marks = [], n = 0;

      hs.forEach(function (h) {
        var sec = isSection(h);
        var li = document.createElement('li');
        if (sec) { li.className = 'is-section'; }
        var key = catOf(h);
        if (key) { li.setAttribute('data-cat', key); }
        var a = document.createElement('a');
        a.href = '#' + h.id;
        if (!sec) { n += 1; }
        a.textContent = h.textContent;
        li.appendChild(a);
        ul.appendChild(li);
        // 高亮挂在 <a> 上：主题既有样式是 `.sidebar-toc a.active`，挂错元素滚动高亮就失效
        marks.push(a);
      });
      toc.appendChild(title);
      toc.appendChild(ul);

      var sidebar = document.querySelector('.sidebar');
      function place() {
        // hide-sidebar（评论区被关掉）时 .sidebar 是 display:none，退回正文内联
        var wantSidebar = !inline && sidebar && getComputedStyle(sidebar).display !== 'none';
        if (wantSidebar) {
          if (toc.parentElement !== sidebar) {
            title.className = 'sidebar-title';
            toc.className = 'sidebar-toc';
            sidebar.insertBefore(toc, sidebar.firstChild);
          }
        } else if (toc.parentElement !== body) {
          title.className = 'post-toc-title';
          toc.className = 'post-toc';
          body.insertBefore(toc, body.firstChild);
        }
      }
      place();

      function onClick(e) {
        var a = e.target.closest('a');
        if (!a) { return; }
        e.preventDefault();
        smoothTo(document.getElementById(a.getAttribute('href').slice(1)));
      }
      function onScroll() { spy(marks); }
      ul.addEventListener('click', onClick);
      window.addEventListener('scroll', onScroll, { passive: true });
      setTimeout(onScroll, 100);

      return {
        kind: 'card',
        reposition: place,
        destroy: function () {
          window.removeEventListener('scroll', onScroll);
          if (toc.parentNode) { toc.parentNode.removeChild(toc); }
        }
      };
    }

    function render() {
      // 先拆旧的再建新的：否则切换断点的一瞬间页面上会有两个目录
      if (view) { view.destroy(); view = null; }
      var inline = window.innerWidth < BREAK_SIDE;
      view = wantFloating() ? buildFloating() : buildCard(inline);
    }

    render();

    var timer = null;
    window.addEventListener('resize', function () {
      // 断点附近来回触发时去抖，避免反复重建 DOM 与监听
      if (timer) { clearTimeout(timer); }
      timer = setTimeout(function () {
        var needFloat = wantFloating();
        if ((view && view.kind === 'float') === needFloat) {
          if (view) { view.reposition(); }
          return;
        }
        render();
      }, 150);
    });

    window.PCYToc = {
      reposition: function () {
        var needFloat = wantFloating();
        if (view && view.kind === 'float') {
          if (needFloat) { view.reposition(); return; }
        } else if (view && !needFloat) {
          view.reposition(); return;
        }
        render();
      }
    };
  };
})();
