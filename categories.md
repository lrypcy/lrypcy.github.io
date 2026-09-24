---
layout: page
title: 分类
---
{% assign graph = site.data.knowledge_graph %}
<p class="page-desc">共 {{ site.categories | size }} 个分类、{{ site.posts | size }} 篇文章。<b>点击任意分类</b>，文章列表会在下方展开（按系列归拢）；再点一次可收起。</p>

  <!-- 分类卡片：按文章数降序（构建时自动排序） -->
  {% assign goods = '' | split: ',' %}
  {% for c in graph.categories %}
    {% assign cnt = site.categories[c.name].size | default: 0 %}
    {% assign nn = cnt | prepend: '00' | slice: -2, 2 %}
    {% assign item = nn | append: ':' | append: c.name %}
    {% assign item_arr = item | split: '|' %}
    {% assign goods = goods | concat: item_arr %}
  {% endfor %}
  {% assign goods = goods | sort | reverse %}

  <div class="category-grid" id="cat-nav">
    {% for item in goods %}
      {% assign parts = item | split: ':' %}
      {% assign cname = parts[1] %}
      {% assign cnt = parts[0] | plus: 0 %}
      {% assign icon = '📁' %}{% assign color = '#94a3b8' %}
      {% for c in graph.categories %}{% if c.name == cname %}{% assign icon = c.icon %}{% assign color = c.color %}{% endif %}{% endfor %}
      <a class="category-card cat-nav-card" href="#{{ cname | url_encode }}" data-cat="{{ cname }}" style="--cat-color: {{ color }}">
        <span class="category-card-icon">{{ icon }}</span>
        <h3 class="category-card-name">{{ cname }}</h3>
        <span class="category-card-count">{{ cnt }} 篇</span>
      </a>
    {% endfor %}
    {% for cat in site.categories %}
    {%- assign cname = cat[0] -%}
    {%- assign is_known = false -%}
    {%- for c in graph.categories -%}{%- if c.name == cname -%}{%- assign is_known = true -%}{%- endif -%}{%- endfor -%}
    {%- unless is_known -%}
      <a class="category-card cat-nav-card" href="#{{ cname | url_encode }}" data-cat="{{ cname }}" style="--cat-color: #94a3b8">
        <span class="category-card-icon">📌</span>
        <h3 class="category-card-name">{{ cname }}</h3>
        <span class="category-card-count">{{ cat[1] | size }} 篇</span>
      </a>
    {%- endunless -%}
    {% endfor %}
  </div>

  <!-- 分类详情：默认不显示，点击卡片后才展开 -->
  <div class="cat-detail" id="cat-detail" hidden>
    <div class="cat-detail-bar">
      <button class="cat-back-btn" id="cat-back" type="button">← 返回全部分类</button>
      <span class="cat-detail-tip">直接点其他分类也可切换</span>
    </div>

    <div id="cat-panels">
      {% for c in graph.categories %}
      {%- assign nseries = 0 -%}
      {%- for s in graph.series -%}{%- if s.category == c.name -%}{%- assign nseries = nseries | plus: 1 -%}{%- endif -%}{%- endfor -%}
      {%- capture claimed -%}{% for s in graph.series %}{% if s.category == c.name %}{% for p in site.posts %}{% assign pp = p.relative_path | default: p.path %}{% assign hit = false %}{% for pre in s.prefixes %}{% if pp contains pre %}{% assign hit = true %}{% endif %}{% endfor %}{% if hit and s.exclude %}{% for ex in s.exclude %}{% if pp contains ex %}{% assign hit = false %}{% endif %}{% endfor %}{% endif %}{% if hit %};{{ p.url }};{% endif %}{% endfor %}{% endif %}{% endfor %}{%- endcapture -%}
      {%- assign orphans = 0 -%}{% for p in site.categories[c.name] %}{% assign key = p.url | prepend: ';' | append: ';' %}{% unless claimed contains key %}{% assign orphans = orphans | plus: 1 %}{% endunless %}{% endfor %}
      <div class="cat-panel" data-cat="{{ c.name }}" style="--cat-color: {{ c.color }}" hidden>
        <div class="cat-panel-head">
          <span class="cat-panel-icon">{{ c.icon }}</span>
          <h3>{{ c.name }}<em>{{ site.categories[c.name].size | default: 0 }} 篇 · {{ nseries }} 个系列</em></h3>
          <p>{{ c.desc }}</p>
        </div>

        {% for s in graph.series %}
        {% if s.category == c.name %}
        {%- capture slist -%}{% for p in site.posts %}{% assign pp = p.relative_path | default: p.path %}{% assign hit = false %}{% for pre in s.prefixes %}{% if pp contains pre %}{% assign hit = true %}{% endif %}{% endfor %}{% if hit and s.exclude %}{% for ex in s.exclude %}{% if pp contains ex %}{% assign hit = false %}{% endif %}{% endfor %}{% endif %}{% if hit %}{{ p.date | date: "%Y-%m-%d" }}@@{{ p.url }}@@{{ p.title }}||{% endif %}{% endfor %}{%- endcapture -%}
        {%- assign sitems = slist | split: '||' | sort -%}
        {%- assign scount = 0 -%}{% for it in sitems %}{% unless it == '' %}{% assign scount = scount | plus: 1 %}{% endunless %}{% endfor %}
        <div class="series-block" id="cat-series-{{ s.id }}">
          <div class="series-head">
            <span class="series-dot" style="background: {{ c.color }}"></span>
            <h4>{{ s.name }}</h4>
            <span class="series-count">{{ scount }} 篇</span>
          </div>
          <p class="series-desc">{{ s.desc }}</p>
          <ol class="series-posts">
            {% for it in sitems %}{% unless it == '' %}{% assign f = it | split: '@@' %}
            <li><a href="{{ f[1] }}">{{ f[2] }}</a><time>{{ f[0] }}</time></li>
            {% endunless %}{% endfor %}
          </ol>
          {% if scount > 8 %}<button class="series-toggle" type="button">展开全部 {{ scount }} 篇 ↓</button>{% endif %}
        </div>
        {% endif %}
        {% endfor %}

        {%- if orphans > 0 -%}
        <div class="series-block series-orphan">
          <div class="series-head">
            <span class="series-dot" style="background: #94a3b8"></span>
            <h4>未归系列的文章</h4>
            <span class="series-count">{{ orphans }} 篇</span>
          </div>
          <ol class="series-posts">
            {% for p in site.categories[c.name] %}{% assign key = p.url | prepend: ';' | append: ';' %}{% unless claimed contains key %}
            <li><a href="{{ p.url }}">{{ p.title }}</a><time>{{ p.date | date: "%Y-%m-%d" }}</time></li>
            {% endunless %}{% endfor %}
          </ol>
        </div>
        {%- endif -%}
      </div>
      {% endfor %}

      {% for cat in site.categories %}
      {%- assign cname = cat[0] -%}
      {%- assign is_known = false -%}
      {%- for c in graph.categories -%}{%- if c.name == cname -%}{%- assign is_known = true -%}{%- endif -%}{%- endfor -%}
      {%- unless is_known -%}
      <div class="cat-panel" data-cat="{{ cname }}" style="--cat-color: #94a3b8" hidden>
        <div class="cat-panel-head">
          <span class="cat-panel-icon">📌</span>
          <h3>{{ cname }}<em>{{ cat[1] | size }} 篇 · 待配置</em></h3>
          <p>这个分类还没写进 <code>_data/knowledge_graph.yml</code>，用兜底样式展示全部文章。</p>
        </div>
        <div class="series-block">
          <div class="series-head">
            <span class="series-dot" style="background: #94a3b8"></span>
            <h4>全部文章</h4>
            <span class="series-count">{{ cat[1] | size }} 篇</span>
          </div>
          <ol class="series-posts">
            {% for p in cat[1] %}
            <li><a href="{{ p.url }}">{{ p.title }}</a><time>{{ p.date | date: "%Y-%m-%d" }}</time></li>
            {% endfor %}
          </ol>
        </div>
      </div>
      {%- endunless -%}
      {% endfor %}
    </div>
  </div>

<noscript>
  <style>
    .cat-detail { display: block !important; }
    .cat-panel[hidden] { display: block !important; }
    .cat-detail-bar { display: none !important; }
  </style>
</noscript>

<script>
(function () {
  var cards  = [].slice.call(document.querySelectorAll('#cat-nav .cat-nav-card'));
  var wrap   = document.getElementById('cat-detail');
  var panels = [].slice.call(document.querySelectorAll('#cat-panels .cat-panel'));
  var back   = document.getElementById('cat-back');
  if (!cards.length || !panels.length || !wrap) { return; }

  function currentCat() {
    var raw = location.hash.replace(/^#/, '');
    if (!raw) { return null; }
    try { return decodeURIComponent(raw.replace(/\+/g, ' ')); } catch (e) { return null; }
  }

  function open(cat, scroll) {
    var hit = false;
    panels.forEach(function (p) {
      var on = p.getAttribute('data-cat') === cat;
      p.hidden = !on;
      if (on) { hit = true; }
    });
    if (!hit) { close(); return; }
    cards.forEach(function (c) { c.classList.toggle('is-active', c.getAttribute('data-cat') === cat); });
    wrap.hidden = false;
    if (scroll && wrap.scrollIntoView) { wrap.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    if (history.replaceState) { history.replaceState(null, '', '#' + encodeURIComponent(cat)); }
  }

  function close() {
    wrap.hidden = true;
    panels.forEach(function (p) { p.hidden = true; });
    cards.forEach(function (c) { c.classList.remove('is-active'); });
    if (history.replaceState) {
      history.replaceState(null, '', location.pathname + location.search);
    }
  }

  cards.forEach(function (c) {
    c.addEventListener('click', function (e) {
      e.preventDefault();
      var cat = c.getAttribute('data-cat');
      // 再点一次同一个分类 = 收起
      if (!wrap.hidden && c.classList.contains('is-active')) { close(); return; }
      open(cat, true);
    });
  });

  if (back) { back.addEventListener('click', close); }

  window.addEventListener('hashchange', function () {
    var cat = currentCat();
    if (cat) { open(cat, false); } else { close(); }
  });

  // 系列内长列表展开/收起
  [].slice.call(document.querySelectorAll('#cat-panels .series-toggle')).forEach(function (btn) {
    btn.setAttribute('data-label', btn.textContent);
    btn.addEventListener('click', function () {
      var blk = btn.closest('.series-block');
      var isOpen = blk.classList.toggle('is-open');
      btn.textContent = isOpen ? '收起 ↑' : btn.getAttribute('data-label');
    });
  });

  var initCat = currentCat();
  if (initCat) { open(initCat, false); }
})();
</script>
