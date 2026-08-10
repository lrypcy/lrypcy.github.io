---
layout: page
title: 分类
---
<div class="post-block page">
  <h1 class="post-title">分类</h1>

  <!-- 分类卡片导航 -->
  <div class="category-grid">
    {% for category in site.categories %}
    <a class="category-card" href="#{{ category[0] | url_encode }}">
      <div class="category-card-icon">📁</div>
      <h3 class="category-card-name">{{ category[0] }}</h3>
      <span class="category-card-count">{{ category[1] | size }} 篇</span>
    </a>
    {% endfor %}
  </div>

  <!-- 按分类分组 -->
  {% for category in site.categories %}
  <div class="category-group">
    <h3 id="{{ category[0] | url_encode }}">{{ category[0] }} <small>({{ category[1] | size }} 篇)</small></h3>
    <ul>
      {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a><time>{{ post.date | date: "%Y-%m-%d" }}</time></li>
      {% endfor %}
    </ul>
  </div>
  {% endfor %}
</div>
