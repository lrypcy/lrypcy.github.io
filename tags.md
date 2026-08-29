---
layout: page
title: 标签
---
<div class="post-block page">
  <h1 class="post-title">标签</h1>

  <!-- 标签云导航 -->
  {% assign tag_max = 0 %}
  {% assign tag_min = 9999 %}
  {% for tag in site.tags %}
    {% assign sz = tag[1] | size %}
    {% if sz > tag_max %}{% assign tag_max = sz %}{% endif %}
    {% if sz < tag_min %}{% assign tag_min = sz %}{% endif %}
  {% endfor %}
  {% assign colors = "#3b82f6,#6366f1,#8b5cf6,#a855f7,#ec4899,#e11d48,#ea580c,#d97706,#16a34a,#0d9488,#0891b2,#1d4ed8" | split: "," %}
  <div class="tag-cloud">
    {% for tag in site.tags %}
      {% assign tag_size = tag[1] | size %}
      {% assign span = tag_max | minus: tag_min %}
      {% if span > 0 %}
        {% assign ratio = tag_size | minus: tag_min | times: 100 | divided_by: span %}
      {% else %}
        {% assign ratio = 50 %}
      {% endif %}
      {% assign font_pct = ratio | plus: 100 %}
      {% assign color_idx = forloop.index0 | modulo: 12 %}
      {% assign color = colors[color_idx] %}
      <a class="tag-item" href="#{{ tag[0] | url_encode }}" style="font-size: {{ font_pct }}%; color: {{ color }};" title="{{ tag_size }} 篇文章">{{ tag[0] }}</a>
    {% endfor %}
  </div>

  <!-- 按标签分组 -->
  {% for tag in site.tags %}
  <div class="tag-group">
    <h3 id="{{ tag[0] | url_encode }}">{{ tag[0] }} <small>({{ tag[1] | size }} 篇)</small></h3>
    <ul>
      {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a><time>{{ post.date | date: "%Y-%m-%d" }}</time></li>
      {% endfor %}
    </ul>
  </div>
  {% endfor %}
</div>
