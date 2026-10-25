---
layout: page
title: 资讯
no_sidebar: true
---
{% assign dailies = site.daily | sort: 'date' | reverse %}
{% if dailies.size == 0 %}
<p class="page-desc">暂无条目。</p>
{% else %}
<div class="news-list" id="newsList">
  {% for note in dailies %}
  <a class="news-row" href="{{ note.url }}">
    <span class="news-date">{{ note.date | date: "%Y-%m-%d" }}</span>
    <span class="news-main">
      <span class="news-title">{{ note.title }}{% if note.rating %}<span class="news-rating" title="本期综合评分 {{ note.rating }}/5">{% assign r = note.rating | plus: 0 %}{% for i in (1..5) %}{% if i <= r %}★{% else %}☆{% endif %}{% endfor %}</span>{% endif %}</span>
      {% if note.summary %}<span class="news-desc">{{ note.summary }}</span>{% endif %}
    </span>
  </a>
  {% endfor %}
</div>
<nav class="pager" id="newsPager" hidden></nav>
<script src="/assets/js/pager.js"></script>
<script>
(function () {
  var items = [].slice.call(document.querySelectorAll('#newsList .news-row'));
  if (!window.pcyPager) { return; }
  function readPage() {
    var v = null;
    try { v = new URLSearchParams(location.search).get('page'); } catch (e) {}
    return parseInt(v, 10) || 1;
  }
  var pager = window.pcyPager({
    items: items,
    perPage: 12,
    container: document.getElementById('newsPager'),
    onChange: function (p) {
      if (history.replaceState) { history.replaceState(null, '', location.pathname + (p > 1 ? '?page=' + p : '')); }
    }
  });
  if (readPage() > 1) { pager.goTo(readPage()); }
  window.addEventListener('popstate', function () { pager.goTo(readPage()); });
})();
</script>
{% endif %}
