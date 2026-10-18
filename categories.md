---
layout: page
title: 分类
---
{% assign graph = site.data.knowledge_graph %}
{% capture known %}{% for sec in site.data.catalog.sections %}{% for c in sec.categories %};{{ c }};{% endfor %}{% endfor %}{% endcapture %}
<!-- 浏览态：全站分类一览（进入单分类页时整体隐藏） -->
<div id="cat-home">
<p class="page-desc">共 {{ site.categories | size }} 个分类、{{ site.posts | size }} 篇文章。<b>点击任意分类</b>，进入按系列归拢的独立页面，地址栏带上 <code>?cat=</code> 参数，可直接分享。</p>

  <!-- 分类卡片：按 _data/catalog.yml 的 AI-Infra 分层排布（与左侧目录同一套顺序） -->
  <div id="cat-nav">
    {% for sec in site.data.catalog.sections %}
    <h3 class="cat-sec-title">{{ sec.name }}<em>{{ sec.hint }}</em></h3>
    <div class="category-grid">
      {% for cname in sec.categories %}
      {%- assign cnt = site.categories[cname].size | default: 0 -%}
      {%- if cnt > 0 -%}
      {%- assign icon = '📁' -%}{%- assign color = '#94a3b8' -%}
      {%- for c in graph.categories -%}{%- if c.name == cname -%}{%- assign icon = c.icon -%}{%- assign color = c.color -%}{%- endif -%}{%- endfor -%}
      <a class="category-card cat-nav-card" href="/categories/?cat={{ cname | url_encode }}" data-cat="{{ cname }}" style="--cat-color: {{ color }}">
        <span class="category-card-icon">{{ icon }}</span>
        <h3 class="category-card-name">{{ cname }}</h3>
        <span class="category-card-count">{{ cnt }} 篇</span>
      </a>
      {%- endif -%}
      {% endfor %}
    </div>
    {% endfor %}

    <!-- 兜底：站点里有、但没写进 catalog.yml 的分类 -->
    {%- assign extra = 0 -%}
    {%- for cat in site.categories -%}{%- assign k = cat[0] | prepend: ';' | append: ';' -%}{%- unless known contains k -%}{%- assign extra = extra | plus: 1 -%}{%- endunless -%}{%- endfor -%}
    {%- if extra > 0 -%}
    <h3 class="cat-sec-title">其他分类<em>未纳入分层</em></h3>
    <div class="category-grid">
      {% for cat in site.categories %}
      {%- assign cname = cat[0] -%}
      {%- assign k = cname | prepend: ';' | append: ';' -%}
      {%- unless known contains k -%}
      <a class="category-card cat-nav-card" href="/categories/?cat={{ cname | url_encode }}" data-cat="{{ cname }}" style="--cat-color: #94a3b8">
        <span class="category-card-icon">📌</span>
        <h3 class="category-card-name">{{ cname }}</h3>
        <span class="category-card-count">{{ cat[1] | size }} 篇</span>
      </a>
      {%- endunless -%}
      {% endfor %}
    </div>
    {%- endif -%}
  </div>
</div>

  <!-- 单分类独立页：URL 带 ?cat= / #分类名 时出现，上方分类卡片整块隐藏 -->
  <div class="cat-detail" id="cat-detail" hidden>
    <div class="cat-detail-bar">
      <button class="cat-back-btn" id="cat-back" type="button">← 返回全部分类</button>
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
        {%- capture slist -%}{% for p in site.posts %}{% assign pp = p.relative_path | default: p.path %}{% assign hit = false %}{% for pre in s.prefixes %}{% if pp contains pre %}{% assign hit = true %}{% endif %}{% endfor %}{% if hit and s.exclude %}{% for ex in s.exclude %}{% if pp contains ex %}{% assign hit = false %}{% endif %}{% endfor %}{% endif %}{% if hit %}{% assign pdate = p.date | date: "%Y-%m-%d" %}{% include post-sortkey.html title=p.title date=pdate url=p.url %}||{% endif %}{% endfor %}{%- endcapture -%}
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
            <li><a href="{{ f[3] }}">{{ f[2] }}</a><time>{{ f[1] }}</time></li>
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

  var home = document.getElementById('cat-home');
  var h1 = document.querySelector('.layout-main .post-title');
  var baseH1 = h1 ? h1.textContent : '';
  var siteTitle = {{ site.title | jsonify }};
  function setDocTitle(t) { document.title = (t || baseH1) + ' | ' + siteTitle; }

  // URL 参数优先 ?cat=，旧式 #分类名 锚点链接继续兼容
  function currentCat() {
    var v = null;
    try { v = new URLSearchParams(location.search).get('cat'); } catch (e) {}
    if (v) { return v; }
    var raw = location.hash.replace(/^#/, '');
    if (!raw) { return null; }
    try { return decodeURIComponent(raw.replace(/\+/g, ' ')); } catch (e) { return null; }
  }

  function setUrl(cat) {
    var url = location.pathname + (cat ? '?cat=' + encodeURIComponent(cat) : '');
    if (history.replaceState) { history.replaceState(null, '', url); }
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
    // 单分类页：上方分类卡片整块让位
    if (home) { home.hidden = true; }
    // 独立页撤掉卡片外框，文章直接铺排（见 .is-filtered 规则）
    document.body.classList.add('is-filtered');
    if (h1) { h1.textContent = baseH1 + '：' + cat; }
    setDocTitle(baseH1 + '：' + cat);
    if (scroll && wrap.scrollIntoView) { wrap.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
    setUrl(cat);
  }

  function close() {
    wrap.hidden = true;
    panels.forEach(function (p) { p.hidden = true; });
    cards.forEach(function (c) { c.classList.remove('is-active'); });
    if (home) { home.hidden = false; }
    document.body.classList.remove('is-filtered');
    if (h1) { h1.textContent = baseH1; }
    setDocTitle(baseH1);
    setUrl(null);
  }

  cards.forEach(function (c) {
    c.addEventListener('click', function (e) {
      e.preventDefault();
      open(c.getAttribute('data-cat'), true);
    });
  });

  if (back) { back.addEventListener('click', close); }

  window.addEventListener('hashchange', function () {
    var cat = currentCat();
    if (cat) { open(cat, false); } else { close(); }
  });

  window.addEventListener('popstate', function () {
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
