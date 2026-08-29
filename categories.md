---
layout: page
title: 分类
---
<div class="post-block page">
  <h1 class="post-title">分类</h1>

  <!-- 分类卡片导航（点击过滤该分类文章） -->
  <div class="category-grid" id="category-nav">
    {% for category in site.categories %}
    {% assign icon = '📁' %}
    {% if category[0] == 'AI Infra' %}{% assign icon = '⚡' %}
    {% elsif category[0] == '强化学习' %}{% assign icon = '🎮' %}
    {% elsif category[0] == '芯片架构' %}{% assign icon = '🖥️' %}
    {% elsif category[0] == '生成模型' %}{% assign icon = '🎨' %}
    {% elsif category[0] == '模型量化' %}{% assign icon = '🎯' %}
    {% elsif category[0] == '深度学习' %}{% assign icon = '🧠' %}
    {% elsif category[0] == '模型蒸馏' %}{% assign icon = '🧪' %}
    {% elsif category[0] == '具身智能' %}{% assign icon = '🤖' %}
    {% elsif category[0] == '编译器技术' %}{% assign icon = '🛠️' %}
    {% elsif category[0] == 'AI 编译器' %}{% assign icon = '🔧' %}
    {% elsif category[0] == '位置编码' %}{% assign icon = '📍' %}
    {% endif %}
    <a class="category-card" data-cat="{{ category[0] }}" href="#{{ category[0] | url_encode }}">
      <span class="category-card-icon">{{ icon }}</span>
      <h3 class="category-card-name">{{ category[0] }}</h3>
      <span class="category-card-count">{{ category[1] | size }} 篇</span>
    </a>
    {% endfor %}
  </div>
  <div id="category-show-all" class="category-show-all" hidden>← 显示全部分类</div>

  <!-- 按分类分组 -->
  <div id="category-list">
  {% for category in site.categories %}
  <div class="category-group" data-cat="{{ category[0] }}">
    <h3 id="{{ category[0] | url_encode }}">{{ category[0] }} <small>({{ category[1] | size }} 篇)</small></h3>
    <ul>
      {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a><time>{{ post.date | date: "%Y-%m-%d" }}</time></li>
      {% endfor %}
    </ul>
  </div>
  {% endfor %}
  </div>
</div>

<script>
(function () {
  var cards = [].slice.call(document.querySelectorAll('#category-nav .category-card'));
  var groups = [].slice.call(document.querySelectorAll('#category-list .category-group'));
  var allBtn = document.getElementById('category-show-all');
  if (!cards.length || !groups.length) { return; }

  function currentCat() {
    var raw = location.hash.replace(/^#/, '');
    if (!raw) { return null; }
    try { return decodeURIComponent(raw.replace(/\+/g, ' ')); } catch (e) { return null; }
  }

  function applyFilter(cat) {
    var showAll = !cat;
    cards.forEach(function (c) {
      c.classList.toggle('is-active', !showAll && c.getAttribute('data-cat') === cat);
    });
    groups.forEach(function (g) {
      g.style.display = (!showAll && g.getAttribute('data-cat') !== cat) ? 'none' : '';
    });
    if (allBtn) { allBtn.hidden = showAll; }
  }

  function scrollToCat(cat) {
    var target = null;
    groups.forEach(function (g) {
      if (g.getAttribute('data-cat') === cat) { target = g; }
    });
    if (target && target.scrollIntoView) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  cards.forEach(function (c) {
    c.addEventListener('click', function (e) {
      e.preventDefault();
      var cat = c.getAttribute('data-cat');
      applyFilter(cat);
      if (history.replaceState) {
        history.replaceState(null, '', '#' + encodeURIComponent(cat));
      }
      scrollToCat(cat);
    });
  });
  if (allBtn) {
    allBtn.addEventListener('click', function (e) {
      e.preventDefault();
      if (history.pushState) {
        history.pushState(null, '', location.pathname + location.search);
      }
      applyFilter(null);
    });
  }
  window.addEventListener('hashchange', function () { applyFilter(currentCat()); });

  var initCat = currentCat();
  applyFilter(initCat);
  if (initCat) { scrollToCat(initCat); }
})();
</script>