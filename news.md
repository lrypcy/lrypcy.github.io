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
      <span class="news-title">{{ note.title }}</span>
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
  if (window.pcyPager) {
    window.pcyPager({ items: items, perPage: 12, container: document.getElementById('newsPager') });
  }
})();
</script>
{% endif %}
