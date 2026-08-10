---
layout: page
title: 归档
---
<div class="post-block archive">
  {% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
  {% for year_group in posts_by_year %}
  <h2 class="archive-year">{{ year_group.name }}</h2>
  <ul class="archive-list">
    {% for post in year_group.items %}
    <li>
      <span class="archive-date">{{ post.date | date: "%m-%d" }}</span>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% endfor %}
  </ul>
  {% endfor %}
</div>
