/* 轻量分页器：pcyPager({ items, perPage, container, onChange })
   items: 元素数组（分页器直接切换其 hidden）；onChange(page) 在翻页后回调 */
(function () {
  window.pcyPager = function (opts) {
    var items = opts.items || [];
    var per = opts.perPage || 20;
    var nav = opts.container;
    var onChange = opts.onChange || function () {};
    var pages = Math.max(1, Math.ceil(items.length / per));
    var cur = 1;

    function renderNav() {
      if (!nav) { return; }
      if (pages <= 1) { nav.hidden = true; nav.innerHTML = ''; return; }
      nav.hidden = false;
      var html = '<button type="button" class="pager-btn pager-prev"' + (cur === 1 ? ' disabled' : '') + '>← 上一页</button>';
      for (var i = 1; i <= pages; i++) {
        html += '<button type="button" class="pager-num' + (i === cur ? ' is-active' : '') + '" data-page="' + i + '">' + i + '</button>';
      }
      html += '<button type="button" class="pager-btn pager-next"' + (cur === pages ? ' disabled' : '') + '>下一页 →</button>';
      html += '<span class="pager-jump">跳至<input type="number" class="pager-jump-input" min="1" max="' + pages + '" value="' + cur + '" aria-label="跳转到指定页">页</span>';
      nav.innerHTML = html;
      [].slice.call(nav.querySelectorAll('button')).forEach(function (b) {
        b.addEventListener('click', function () {
          if (b.hasAttribute('data-page')) { api.goTo(parseInt(b.getAttribute('data-page'), 10)); }
          else if (b.classList.contains('pager-prev')) { api.goTo(cur - 1); }
          else { api.goTo(cur + 1); }
        });
      });
      var jump = nav.querySelector('.pager-jump-input');
      if (jump) {
        var doJump = function () {
          var v = parseInt(jump.value, 10);
          if (!isNaN(v)) { api.goTo(v); }
        };
        jump.addEventListener('keydown', function (e) { if (e.key === 'Enter') { doJump(); } });
        jump.addEventListener('change', doJump);
      }
    }

    var api = {
      page: function () { return cur; },
      pageCount: function () { return pages; },
      goTo: function (p) {
        cur = Math.min(pages, Math.max(1, p || 1));
        items.forEach(function (el, idx) {
          el.hidden = !(idx >= (cur - 1) * per && idx < cur * per);
        });
        renderNav();
        onChange(cur);
      }
    };
    api.goTo(1);
    return api;
  };
})();
