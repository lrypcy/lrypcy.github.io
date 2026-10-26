---
layout: page
title: 资讯
no_sidebar: true
---
{% assign dailies = site.daily | sort: 'date' | reverse %}
{% if dailies.size == 0 %}
<p class="page-desc">暂无条目。</p>
{% else %}
{% comment %}
  分类不单独占一排筛选按钮 —— 每期自己的标签就是入口（词表在
  _data/daily_categories.yml，key 在前端只做映射，中文 label 由它给）。
  点某期的标签 = 只看该分类的各期；这行提示条只在筛选态出现，平时隐藏。
{% endcomment %}
<div class="news-filter" id="newsFilter" hidden>
  <span class="news-filter-label">正在查看</span>
  <span class="news-filter-cat" id="newsFilterCat"></span>
  <button type="button" class="news-clear" id="newsClear">看全部 ×</button>
</div>

<div class="news-list" id="newsList">
  {% for note in dailies %}
  <a class="news-row" href="{{ note.url }}" data-cats="{{ note.cats | join: ',' }}">
    <span class="news-date">{{ note.date | date: "%Y-%m-%d" }}</span>
    <span class="news-main">
      <span class="news-title">{{ note.title }}{% if note.rating %}<span class="news-rating" title="本期综合评分 {{ note.rating }}/5">{% assign r = note.rating | plus: 0 %}{% for i in (1..5) %}{% if i <= r %}★{% else %}☆{% endif %}{% endfor %}</span>{% endif %}</span>
      {% if note.summary %}<span class="news-desc">{{ note.summary }}</span>{% endif %}
      {% if note.cats %}<span class="news-cats">{% for cat in site.data.daily_categories %}{% if note.cats contains cat.key %}<span class="news-cat" data-cat="{{ cat.key }}" data-label="{{ cat.label }}" title="只看「{{ cat.label }}」的各期" role="button" tabindex="0">{{ cat.label }}</span>{% endif %}{% endfor %}</span>{% endif %}
    </span>
  </a>
  {% endfor %}
</div>
<p class="news-empty" id="newsEmpty" hidden>该分类下暂无条目。</p>
<nav class="pager" id="newsPager" hidden></nav>
<script src="/assets/js/pager.js"></script>
<script>
/* 列表交互：分类筛选（?cat=<key>）+ 分页（?page=N）共存。
   要点：
   1. 每次筛选先把所有行 hidden=false，再用筛选后的数组重建 pcyPager，
      否则分页器只认识自己那一批 items，翻页后会残留上一次的隐藏行。
   2. data-cats 是**逗号**连接的（Liquid join: ','），判断包含必须也用逗号包围。
      早先按 | 包围去查，永远查不到 → 任何分类都是空集。 */
(function () {
  var list = document.getElementById('newsList');
  var rows = [].slice.call(list.querySelectorAll('.news-row'));
  var nav = document.getElementById('newsPager');
  var bar = document.getElementById('newsFilter');
  var barCat = document.getElementById('newsFilterCat');
  var clearBtn = document.getElementById('newsClear');
  var empty = document.getElementById('newsEmpty');
  var rowChips = [].slice.call(list.querySelectorAll('.news-cat'));

  function readParam(name) {
    try { return new URLSearchParams(location.search).get(name) || ''; } catch (e) { return ''; }
  }
  function readPage() { return parseInt(readParam('page'), 10) || 1; }

  /* 只认列表里真实出现过的分类，?cat= 传了别的（或手敲错的）就退回「全部」 */
  function normalizeCat(c) {
    if (!c) { return ''; }
    for (var i = 0; i < rowChips.length; i++) {
      if (rowChips[i].getAttribute('data-cat') === c) { return c; }
    }
    return '';
  }

  function match(r, key) {
    if (!key) { return true; }
    return (',' + (r.getAttribute('data-cats') || '') + ',').indexOf(',' + key + ',') >= 0;
  }

  function syncUrl(cat, p) {
    if (!history.replaceState) { return; }
    var q = [];
    if (cat) { q.push('cat=' + encodeURIComponent(cat)); }
    if (p > 1) { q.push('page=' + p); }
    history.replaceState(null, '', location.pathname + (q.length ? '?' + q.join('&') : ''));
  }

  function apply(cat, page) {
    cat = normalizeCat(cat);
    rows.forEach(function (r) { r.hidden = false; });
    var subset = rows.filter(function (r) { return match(r, cat); });
    rows.forEach(function (r) { if (subset.indexOf(r) < 0) { r.hidden = true; } });
    if (empty) { empty.hidden = subset.length > 0; }
    if (bar) {
      bar.hidden = !cat;
      if (cat && barCat) {
        var label = cat;
        rowChips.some(function (c) {
          if (c.getAttribute('data-cat') === cat) { label = c.getAttribute('data-label') || cat; return true; }
          return false;
        });
        barCat.textContent = '「' + label + '」';
      }
    }
    rowChips.forEach(function (c) {
      var on = c.getAttribute('data-cat') === cat;
      if (on) { c.classList.add('is-active'); } else { c.classList.remove('is-active'); }
      c.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    if (!window.pcyPager) { return; }
    var pager = window.pcyPager({
      items: subset,
      perPage: 12,
      container: nav,
      onChange: function (p) { syncUrl(cat, p); }
    });
    if (page > 1) { pager.goTo(page); }
  }

  /* 行内标签：点它只筛选，不进详情页（.news-row 本身是 <a>，这里拦掉默认跳转） */
  rowChips.forEach(function (c) {
    function pick(e) {
      e.preventDefault();
      e.stopPropagation();
      var key = c.getAttribute('data-cat');
      var on = c.classList.contains('is-active');
      apply(on ? '' : key, 1);
    }
    c.addEventListener('click', pick);
    c.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { pick(e); }
    });
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', function () { apply('', 1); });
  }

  apply(readParam('cat'), readPage());
  window.addEventListener('popstate', function () { apply(readParam('cat'), readPage()); });
})();
</script>
{% endif %}
