---
layout: page
title: 文章
no_sidebar: true
no_title: true
---
{% capture known %}{% for sec in site.data.catalog.sections %}{% for c in sec.categories %};{{ c }};{% endfor %}{% endfor %}{% endcapture %}
{%- comment -%} 单分类页「上一个 / 下一个」用的全站分类顺序：catalog 分层在前，兜底分类殿后 {%- endcomment -%}
{%- assign cat_order = '' | split: ',' -%}
{%- for sec in site.data.catalog.sections -%}{%- for cname in sec.categories -%}{%- if site.categories[cname].size > 0 -%}{%- assign cat_order = cat_order | push: cname -%}{%- endif -%}{%- endfor -%}{%- endfor -%}
{%- for cat in site.categories -%}{%- assign k = cat[0] | prepend: ';' | append: ';' -%}{%- unless known contains k -%}{%- assign cat_order = cat_order | push: cat[0] -%}{%- endunless -%}{%- endfor -%}
{% assign graph = site.data.knowledge_graph %}
{% capture tag_sort %}{% for tag in site.tags %}{% assign n = tag[1] | size %}{{ n | prepend: '0000' | slice: -4, 4 }}@@{{ tag[0] }}@@{{ n }}||{% endfor %}{% endcapture %}
{% assign tag_items = tag_sort | split: '||' | sort | reverse %}
{% assign hot_tags = 0 %}
{% for tag in site.tags %}{% assign n = tag[1] | size %}{% if n >= 2 %}{% assign hot_tags = hot_tags | plus: 1 %}{% endif %}{% endfor %}

<div class="page-tabs" id="page-tabs">
  <button class="page-tab is-active" data-tab="cats" type="button">目录树</button>
  <button class="page-tab" data-tab="tags" type="button">标签</button>
  <button class="page-tab" data-tab="archives" type="button">文章</button>
</div>

<!-- ======================= Tab 1：目录树（分类） ======================= -->
<div id="tab-cats" class="tab-panel">
<div id="cat-home">

  <div id="cat-nav">
    {% for sec in site.data.catalog.sections %}
    <h3 class="cat-sec-title">{{ sec.name }}<em>{{ sec.hint }}</em></h3>
    <div class="cat-tree">
      {% for cname in sec.categories %}
      {%- assign cnt = site.categories[cname].size | default: 0 -%}
      {%- if cnt > 0 -%}
      {%- assign icon = '📁' -%}{%- assign cdesc = '' -%}{%- assign nseries = 0 -%}
      {%- for c in graph.categories -%}{%- if c.name == cname -%}{%- assign icon = c.icon -%}{%- assign cdesc = c.desc -%}{%- endif -%}{%- endfor -%}
      {%- for s in graph.series -%}{%- if s.category == cname -%}{%- assign nseries = nseries | plus: 1 -%}{%- endif -%}{%- endfor -%}
      <a class="cat-row" href="/posts/?cat={{ cname | url_encode }}" data-cat="{{ cname }}">
        <span class="cat-row-icon">{{ icon }}</span>
        <span class="cat-row-name">{{ cname }}</span>
        <span class="cat-row-desc">{{ cdesc | strip }}</span>
        <span class="cat-row-meta">{{ cnt }} 篇{% if nseries > 0 %} · {{ nseries }} 系列{% endif %}</span>
      </a>
      {%- endif -%}
      {% endfor %}
    </div>
    {% endfor %}

    {%- assign extra = 0 -%}
    {%- for cat in site.categories -%}{%- assign k = cat[0] | prepend: ';' | append: ';' -%}{%- unless known contains k -%}{%- assign extra = extra | plus: 1 -%}{%- endunless -%}{%- endfor -%}
    {%- if extra > 0 -%}
    <h3 class="cat-sec-title">其他分类<em>未纳入分层</em></h3>
    <div class="cat-tree">
      {% for cat in site.categories %}
      {%- assign cname = cat[0] -%}
      {%- assign k = cname | prepend: ';' | append: ';' -%}
      {%- unless known contains k -%}
      <a class="cat-row" href="/posts/?cat={{ cname | url_encode }}" data-cat="{{ cname }}">
        <span class="cat-row-icon">📌</span>
        <span class="cat-row-name">{{ cname }}</span>
        <span class="cat-row-desc">未纳入分层，点击查看全部文章</span>
        <span class="cat-row-meta">{{ cat[1] | size }} 篇</span>
      </a>
      {%- endunless -%}
      {% endfor %}
    </div>
    {%- endif -%}
  </div>
</div>

  <div class="cat-detail" id="cat-detail" hidden>
    <div class="cat-detail-bar">
      <button class="cat-back-btn" id="cat-back" type="button">← 返回目录树</button>
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
        {%- assign scount = 0 -%}{% for it in sitems %}{%- unless it == '' -%}{%- assign scount = scount | plus: 1 -%}{%- endunless -%}{%- endfor %}
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

        {%- assign myidx = -1 -%}
        {%- for o in cat_order -%}{%- if o == c.name -%}{%- assign myidx = forloop.index0 -%}{%- endif -%}{%- endfor -%}
        {%- if myidx >= 0 -%}
        {%- assign pn_prev = '' -%}{%- assign pn_next = '' -%}
        {%- if myidx > 0 -%}{%- assign pn_i = myidx | minus: 1 -%}{%- assign pn_prev = cat_order[pn_i] -%}{%- endif -%}
        {%- assign pn_last = cat_order.size | minus: 1 -%}
        {%- if myidx < pn_last -%}{%- assign pn_i = myidx | plus: 1 -%}{%- assign pn_next = cat_order[pn_i] -%}{%- endif -%}
        {%- if pn_prev != '' or pn_next != '' -%}
        <nav class="cat-pn">
          {%- if pn_prev != '' -%}<a class="cat-pn-link prev" href="/posts/?cat={{ pn_prev | url_encode }}" data-pn="{{ pn_prev }}">← {{ pn_prev }}</a>{%- else -%}<span></span>{%- endif -%}
          {%- if pn_next != '' -%}<a class="cat-pn-link next" href="/posts/?cat={{ pn_next | url_encode }}" data-pn="{{ pn_next }}">{{ pn_next }} →</a>{%- else -%}<span></span>{%- endif -%}
        </nav>
        {%- endif -%}
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

        {%- assign myidx = -1 -%}
        {%- for o in cat_order -%}{%- if o == cname -%}{%- assign myidx = forloop.index0 -%}{%- endif -%}{%- endfor -%}
        {%- if myidx >= 0 -%}
        {%- assign pn_prev = '' -%}{%- assign pn_next = '' -%}
        {%- if myidx > 0 -%}{%- assign pn_i = myidx | minus: 1 -%}{%- assign pn_prev = cat_order[pn_i] -%}{%- endif -%}
        {%- assign pn_last = cat_order.size | minus: 1 -%}
        {%- if myidx < pn_last -%}{%- assign pn_i = myidx | plus: 1 -%}{%- assign pn_next = cat_order[pn_i] -%}{%- endif -%}
        {%- if pn_prev != '' or pn_next != '' -%}
        <nav class="cat-pn">
          {%- if pn_prev != '' -%}<a class="cat-pn-link prev" href="/posts/?cat={{ pn_prev | url_encode }}" data-pn="{{ pn_prev }}">← {{ pn_prev }}</a>{%- else -%}<span></span>{%- endif -%}
          {%- if pn_next != '' -%}<a class="cat-pn-link next" href="/posts/?cat={{ pn_next | url_encode }}" data-pn="{{ pn_next }}">{{ pn_next }} →</a>{%- else -%}<span></span>{%- endif -%}
        </nav>
        {%- endif -%}
        {%- endif -%}
      </div>
      {%- endunless -%}
      {% endfor %}
    </div>
  </div>
</div>

<!-- ======================= Tab 2：标签 ======================= -->
<div id="tab-tags" class="tab-panel" hidden>
  <div id="tag-nav">
    <p class="page-desc">共 {{ site.tags | size }} 个标签、{{ site.posts | size }} 篇文章。默认显示被 2 篇以上文章使用的 {{ hot_tags }} 个标签，可搜索或展开全部。</p>

    <div class="tag-tools">
      <label class="tag-search">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="M20 20l-4.2-4.2"></path></svg>
        <input id="tagSearch" type="search" autocomplete="off" placeholder="搜索标签…">
      </label>
      <span class="tag-count-info" id="tagCountInfo"></span>
    </div>

    <div class="tag-cloud" id="tagCloud">
      {% for it in tag_items %}{% unless it == '' %}{% assign f = it | split: '@@' %}{% assign n = f[2] | plus: 0 %}
      <a class="tag-item" href="/posts/?tag={{ f[1] | url_encode }}" data-tag="{{ f[1] }}" data-count="{{ n }}"{% if n < 2 %} data-low="1"{% endif %}><span class="tag-name">{{ f[1] }}</span><span class="tag-count">{{ n }}</span></a>
      {% endunless %}{% endfor %}
    </div>

    <button class="tag-show-all" id="tagShowAll" type="button">显示全部 {{ site.tags | size }} 个标签 ↓</button>
  </div>

  <div class="tag-filter-bar" id="tag-filter-bar" hidden>
    <button class="tag-back-btn" id="tag-back" type="button">← 返回全部标签</button>
    <span class="tag-filter-current"><b id="tag-filter-name"></b><em id="tag-filter-count"></em></span>
  </div>

  {% for tag in site.tags %}
  <div class="tag-group" data-tag="{{ tag[0] }}">
    <h3 id="{{ tag[0] | url_encode }}">{{ tag[0] }} <small>({{ tag[1] | size }} 篇)</small></h3>
    <ul>
      {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a><time>{{ post.date | date: "%Y-%m-%d" }}</time></li>
      {% endfor %}
    </ul>
  </div>
  {% endfor %}
</div>

<!-- ======================= Tab 3：文章（归档） ======================= -->
<div id="tab-archives" class="tab-panel" hidden>
<div class="archives" id="archivesList">
  {% assign posts_by_year = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
  {% for year_group in posts_by_year %}
  <section class="archive-year-block">
    <h2 class="archive-year">{{ year_group.name }}<span>{{ year_group.items | size }} 篇</span></h2>
    <ul class="archive-list">
      {% for post in year_group.items %}
      <li>
        <span class="archive-date">{{ post.date | date: "%m-%d" }}</span>
        <a href="{{ post.url }}">{{ post.title }}</a>
      </li>
      {% endfor %}
    </ul>
  </section>
  {% endfor %}
</div>
<nav class="pager" id="archivesPager" hidden></nav>
</div>

<noscript>
  <style>
    .tab-panel[hidden] { display: block !important; }
    .page-tabs { display: none !important; }
    .pager { display: none !important; }
  </style>
</noscript>

<script src="/assets/js/pager.js"></script>
<script>
(function () {
  var TABS = ['cats', 'tags', 'archives'];
  var tabBtns = [].slice.call(document.querySelectorAll('#page-tabs .page-tab'));
  var tabPanels = {};
  TABS.forEach(function (t) { tabPanels[t] = document.getElementById('tab-' + t); });
  if (!tabBtns.length || !tabPanels.cats) { return; }
  var siteTitle = {{ site.title | jsonify }};

  var state = { tab: 'cats', cat: null, tag: null, page: 1 };
  var archPager = null;

  function readState() {
    var p = null;
    try { p = new URLSearchParams(location.search); } catch (e) {}
    function q(k) { try { return p ? p.get(k) : null; } catch (e) { return null; } }
    var s = { tab: q('tab') || 'cats', cat: q('cat') || null, tag: q('tag') || null, page: parseInt(q('page'), 10) || 1 };
    if (s.cat) { s.tab = 'cats'; }
    if (s.tag) { s.tab = 'tags'; }
    if (TABS.indexOf(s.tab) < 0) { s.tab = 'cats'; }
    // 旧式 #标签名 锚点链接兼容
    if (!s.tag && location.hash) {
      var raw = location.hash.replace(/^#/, '');
      if (raw) { try { s.tag = decodeURIComponent(raw.replace(/\+/g, ' ')); s.tab = 'tags'; } catch (e) {} }
    }
    return s;
  }

  function syncUrl() {
    var p = [];
    if (state.tab !== 'cats') { p.push('tab=' + state.tab); }
    if (state.tab === 'cats' && state.cat) { p.push('cat=' + encodeURIComponent(state.cat)); }
    if (state.tab === 'tags' && state.tag) { p.push('tag=' + encodeURIComponent(state.tag)); }
    if (state.tab === 'archives' && state.page > 1) { p.push('page=' + state.page); }
    var qs = p.join('&');
    if (history.replaceState) { history.replaceState(null, '', location.pathname + (qs ? '?' + qs : '')); }
  }

  function scrollTop() { window.scrollTo({ top: 0, behavior: 'smooth' }); }

  /* ---- 目录树（分类）---- */
  var catHome = document.getElementById('cat-home');
  var catWrap = document.getElementById('cat-detail');
  var catPanels = [].slice.call(document.querySelectorAll('#cat-panels .cat-panel'));
  var catCards = [].slice.call(document.querySelectorAll('#cat-nav .cat-row'));

  function catPanelOf(cat) {
    for (var i = 0; i < catPanels.length; i++) {
      if (catPanels[i].getAttribute('data-cat') === cat) { return catPanels[i]; }
    }
    return null;
  }

  function renderCats() {
    var g = state.cat ? catPanelOf(state.cat) : null;
    if (state.cat && !g) { state.cat = null; }
    var solo = !!g;
    if (catHome) { catHome.hidden = solo; }
    if (catWrap) { catWrap.hidden = !solo; }
    catPanels.forEach(function (p) { p.hidden = !solo || p !== g; });
    catCards.forEach(function (c) { c.classList.toggle('is-active', solo && c.getAttribute('data-cat') === state.cat); });
    return solo;
  }

  /* ---- 标签 ---- */
  var tagNav = document.getElementById('tag-nav');
  var tagBar = document.getElementById('tag-filter-bar');
  var tagGroups = [].slice.call(document.querySelectorAll('.tag-group'));
  var tagCloud = document.querySelector('.tag-cloud');
  var tagNameEl = document.getElementById('tag-filter-name');
  var tagCountEl = document.getElementById('tag-filter-count');

  function tagGroupOf(tag) {
    for (var i = 0; i < tagGroups.length; i++) {
      if (tagGroups[i].getAttribute('data-tag') === tag) { return tagGroups[i]; }
    }
    return null;
  }

  function renderTags() {
    var g = state.tag ? tagGroupOf(state.tag) : null;
    if (state.tag && !g) { state.tag = null; }
    var solo = !!g;
    tagGroups.forEach(function (el) {
      el.hidden = !solo || el !== g;
      el.classList.toggle('is-solo', solo && el === g);
    });
    if (tagCloud) {
      [].slice.call(tagCloud.querySelectorAll('.tag-item')).forEach(function (a) {
        a.classList.toggle('is-active', solo && a.getAttribute('data-tag') === state.tag);
      });
    }
    if (tagNav) { tagNav.hidden = solo; }
    if (tagBar) { tagBar.hidden = !solo; }
    if (solo) {
      if (tagNameEl) { tagNameEl.textContent = state.tag; }
      if (tagCountEl) { tagCountEl.textContent = '（' + g.querySelectorAll('li').length + ' 篇）'; }
    }
    return solo;
  }

  /* ---- tab 与全局渲染 ---- */
  function renderTabs() {
    tabBtns.forEach(function (b) { b.classList.toggle('is-active', b.getAttribute('data-tab') === state.tab); });
    TABS.forEach(function (t) { if (tabPanels[t]) { tabPanels[t].hidden = state.tab !== t; } });
  }

  function render() {
    var catSolo = renderCats();
    var tagSolo = renderTags();
    renderTabs();
    if (archPager && archPager.page() !== state.page) { archPager.goTo(state.page); }
    document.body.classList.toggle('is-filtered', (state.tab === 'cats' && catSolo) || (state.tab === 'tags' && tagSolo));
    var suffix = (state.tab === 'cats' && state.cat) ? '：' + state.cat : ((state.tab === 'tags' && state.tag) ? '：' + state.tag : '');
    document.title = '文章' + suffix + ' | ' + siteTitle;
  }

  /* ---- 事件 ---- */
  tabBtns.forEach(function (b) {
    b.addEventListener('click', function () {
      state.tab = b.getAttribute('data-tab');
      syncUrl(); render(); scrollTop();
    });
  });

  catCards.forEach(function (c) {
    c.addEventListener('click', function (e) {
      e.preventDefault();
      state.cat = c.getAttribute('data-cat'); state.tab = 'cats';
      syncUrl(); render(); scrollTop();
    });
  });

  var catBack = document.getElementById('cat-back');
  if (catBack) {
    catBack.addEventListener('click', function () {
      state.cat = null; syncUrl(); render(); scrollTop();
    });
  }

  [].slice.call(document.querySelectorAll('#cat-panels .cat-pn-link')).forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      state.cat = a.getAttribute('data-pn');
      syncUrl(); render(); scrollTop();
    });
  });

  if (tagCloud) {
    tagCloud.addEventListener('click', function (e) {
      var a = e.target && e.target.closest ? e.target.closest('.tag-item') : null;
      if (!a) { return; }
      e.preventDefault();
      state.tag = a.getAttribute('data-tag'); state.tab = 'tags';
      syncUrl(); render(); scrollTop();
    });
  }

  var tagBack = document.getElementById('tag-back');
  if (tagBack) {
    tagBack.addEventListener('click', function () {
      state.tag = null; syncUrl(); render(); scrollTop();
    });
  }

  window.addEventListener('popstate', function () { state = readState(); render(); });
  window.addEventListener('hashchange', function () { state = readState(); syncUrl(); render(); });

  /* ---- 单分类面板：系列展开/收起 ---- */
  [].slice.call(document.querySelectorAll('#cat-panels .series-toggle')).forEach(function (btn) {
    btn.setAttribute('data-label', btn.textContent);
    btn.addEventListener('click', function () {
      var blk = btn.closest('.series-block');
      var isOpen = blk.classList.toggle('is-open');
      btn.textContent = isOpen ? '收起 ↑' : btn.getAttribute('data-label');
    });
  });

  /* ---- 标签搜索 + 低频折叠 ---- */
  var search = document.getElementById('tagSearch');
  var countInfo = document.getElementById('tagCountInfo');
  var showAllBtn = document.getElementById('tagShowAll');
  var cloudItems = tagCloud ? [].slice.call(tagCloud.querySelectorAll('.tag-item')) : [];
  var showAll = false;

  function renderCloud() {
    if (!cloudItems.length) { return; }
    var q = ((search && search.value) || '').trim().toLowerCase();
    var shown = 0;
    cloudItems.forEach(function (a) {
      var name = (a.getAttribute('data-tag') || '').toLowerCase();
      var low = a.getAttribute('data-low') === '1';
      var ok = (!q || name.indexOf(q) >= 0) && (!!q || showAll || !low);
      a.hidden = !ok;
      if (ok) { shown++; }
    });
    if (countInfo) {
      countInfo.textContent = q ? '匹配 ' + shown + ' 个' : (showAll ? '全部 ' + cloudItems.length + ' 个' : '常用 ' + shown + ' 个');
    }
    if (showAllBtn) {
      showAllBtn.hidden = !!q;
      showAllBtn.textContent = showAll ? '收起低频标签 ↑' : '显示全部 ' + cloudItems.length + ' 个标签 ↓';
    }
  }
  if (search) { search.addEventListener('input', renderCloud); }
  if (showAllBtn) {
    showAllBtn.addEventListener('click', function () { showAll = !showAll; renderCloud(); });
  }
  renderCloud();

  /* ---- 归档分页：?page= 深链 + 翻页时隐藏没有可见条目的年份块 ---- */
  function updateYearBlocks() {
    [].slice.call(document.querySelectorAll('#tab-archives .archive-year-block')).forEach(function (blk) {
      blk.hidden = blk.querySelectorAll('.archive-list li:not([hidden])').length === 0;
    });
  }
  var archItems = [].slice.call(document.querySelectorAll('#tab-archives .archive-list li'));
  var pagerReady = false;
  if (window.pcyPager && archItems.length) {
    archPager = window.pcyPager({
      items: archItems,
      perPage: 30,
      container: document.getElementById('archivesPager'),
      onChange: function (p) {
        updateYearBlocks();
        if (!pagerReady) { return; }
        state.page = p;
        syncUrl();
      }
    });
    pagerReady = true;
  }

  state = readState();
  if (archPager && state.page !== archPager.page()) { archPager.goTo(state.page); }
  syncUrl();
  render();
})();
</script>
