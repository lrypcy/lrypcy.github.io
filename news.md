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
  分类词表在 _data/daily_categories.yml（唯一事实源），这里只做 key → label 映射：
  遍历词表（顺序即展示顺序），判断某期 note.cats 是否 contains 该 key。
  筛选条只渲染「现有日报里出现过」的分类，没用到的分类不出现。
{% endcomment %}
{% assign used_n = 0 %}
{% assign used_keys = '' %}
{% for cat in site.data.daily_categories %}
  {% assign hit = 0 %}
  {% for note in dailies %}
    {% if note.cats %}
      {% if note.cats contains cat.key %}{% assign hit = hit | plus: 1 %}{% endif %}
    {% endif %}
  {% endfor %}
  {% if hit > 0 %}
    {% assign used_n = used_n | plus: 1 %}
    {% if used_keys == '' %}
      {% assign used_keys = cat.key %}
    {% else %}
      {% assign used_keys = used_keys | append: ',' | append: cat.key %}
    {% endif %}
  {% endif %}
{% endfor %}
{% assign used_cat_list = used_keys | split: ',' %}

{% if used_n > 0 %}
<div class="news-filter" id="newsFilter">
  <span class="news-filter-label">分类</span>
  <button type="button" class="news-chip is-active" data-cat="">全部</button>
  {% for cat in site.data.daily_categories %}
    {% if used_cat_list contains cat.key %}
  <button type="button" class="news-chip" data-cat="{{ cat.key }}">{{ cat.label }}</button>
    {% endif %}
  {% endfor %}
</div>
{% endif %}

<div class="news-list" id="newsList">
  {% for note in dailies %}
  <a class="news-row" href="{{ note.url }}" data-cats="{{ note.cats | join: ',' }}">
    <span class="news-date">{{ note.date | date: "%Y-%m-%d" }}</span>
    <span class="news-main">
      <span class="news-title">{{ note.title }}{% if note.rating %}<span class="news-rating" title="本期综合评分 {{ note.rating }}/5">{% assign r = note.rating | plus: 0 %}{% for i in (1..5) %}{% if i <= r %}★{% else %}☆{% endif %}{% endfor %}</span>{% endif %}{% if note.cats %}<span class="news-cats">{% for cat in site.data.daily_categories %}{% if note.cats contains cat.key %}<span class="news-cat" data-cat="{{ cat.key }}" title="只看「{{ cat.label }}」的各期" role="button" tabindex="0">{{ cat.label }}</span>{% endif %}{% endfor %}</span>{% endif %}</span>
      {% if note.summary %}<span class="news-desc">{{ note.summary }}</span>{% endif %}
    </span>
  </a>
  {% endfor %}
</div>
<p class="news-empty" id="newsEmpty" hidden>该分类下暂无条目。</p>
<nav class="pager" id="newsPager" hidden></nav>
<script src="/assets/js/pager.js"></script>
<script>
/* 列表交互：分类筛选（?cat=<key>）+ 分页（?page=N）共存。
   要点：每次筛选先把所有行 hidden=false，再用筛选后的数组重建 pcyPager，
   否则分页器只认识自己那一批 items，翻页后会残留上一次的隐藏行。 */
(function () {
  var list = document.getElementById('newsList');
  var rows = [].slice.call(list.querySelectorAll('.news-row'));
  var nav = document.getElementById('newsPager');
  var bar = document.getElementById('newsFilter');
  var empty = document.getElementById('newsEmpty');
  var chips = bar ? [].slice.call(bar.querySelectorAll('.news-chip')) : [];
  var rowChips = [].slice.call(list.querySelectorAll('.news-cat'));

  function readParam(name) {
    try { return new URLSearchParams(location.search).get(name) || ''; } catch (e) { return ''; }
  }
  function readPage() { return parseInt(readParam('page'), 10) || 1; }

  /* 只认筛选条里出现过的分类，?cat= 传了别的就退回「全部」 */
  function normalizeCat(c) {
    for (var i = 0; i < chips.length; i++) {
      if (chips[i].getAttribute('data-cat') === c) { return c; }
    }
    return '';
  }

  function match(r, key) {
    if (!key) { return true; }
    return ('|' + (r.getAttribute('data-cats') || '') + '|').indexOf('|' + key + '|') >= 0;
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
    chips.forEach(function (c) {
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

  chips.forEach(function (c) {
    c.addEventListener('click', function () { apply(c.getAttribute('data-cat'), 1); });
  });

  /* 行内的分类 chip：点它只筛选，不进详情页（.news-row 本身是 <a>，这里拦掉默认跳转） */
  rowChips.forEach(function (c) {
    function pick(e) {
      e.preventDefault();
      e.stopPropagation();
      apply(c.getAttribute('data-cat'), 1);
      if (bar) { bar.scrollIntoView({ block: 'start' }); }
    }
    c.addEventListener('click', pick);
    c.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { pick(e); }
    });
  });

  apply(readParam('cat'), readPage());
  window.addEventListener('popstate', function () { apply(readParam('cat'), readPage()); });
})();
</script>
{% endif %}
