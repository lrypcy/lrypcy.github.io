---
layout: page
title: 标签
no_sidebar: true
---
{% capture tag_sort %}{% for tag in site.tags %}{% assign n = tag[1] | size %}{{ n | prepend: '0000' | slice: -4, 4 }}@@{{ tag[0] }}@@{{ n }}||{% endfor %}{% endcapture %}
{% assign tag_items = tag_sort | split: '||' | sort | reverse %}
{% assign hot_tags = 0 %}
{% for tag in site.tags %}{% assign n = tag[1] | size %}{% if n >= 2 %}{% assign hot_tags = hot_tags | plus: 1 %}{% endif %}{% endfor %}

  <!-- 浏览态：全站标签一览（进入单标签页时整体隐藏） -->
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
      <a class="tag-item" href="/tags/?tag={{ f[1] | url_encode }}" data-tag="{{ f[1] }}" data-count="{{ n }}"{% if n < 2 %} data-low="1"{% endif %}><span class="tag-name">{{ f[1] }}</span><span class="tag-count">{{ n }}</span></a>
      {% endunless %}{% endfor %}
    </div>

    <button class="tag-show-all" id="tagShowAll" type="button">显示全部 {{ site.tags | size }} 个标签 ↓</button>
  </div>

  <!-- 单标签独立页：URL 带 ?tag= / #tag 时出现 -->
  <div class="tag-filter-bar" id="tag-filter-bar" hidden>
    <button class="tag-back-btn" id="tag-back" type="button">← 返回全部标签</button>
    <span class="tag-filter-current"><b id="tag-filter-name"></b><em id="tag-filter-count"></em></span>
  </div>

  <!-- 按标签分组：JS 按 URL 参数只保留命中的一组 -->
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

<script>
(function () {
  var cloud = document.querySelector('.tag-cloud');
  var nav = document.getElementById('tag-nav');
  var groups = [].slice.call(document.querySelectorAll('.tag-group'));
  var bar = document.getElementById('tag-filter-bar');
  var back = document.getElementById('tag-back');
  var nameEl = document.getElementById('tag-filter-name');
  var countEl = document.getElementById('tag-filter-count');
  var h1 = document.querySelector('.layout-main .post-title');
  if (!groups.length) { return; }
  var baseH1 = h1 ? h1.textContent : '';
  var siteTitle = {{ site.title | jsonify }};

  function setDocTitle(t) {
    document.title = (t || baseH1) + ' | ' + siteTitle;
  }

  // URL 参数优先 ?tag=，旧式 #tag 锚点链接兼容
  function currentTag() {
    var v = null;
    try { v = new URLSearchParams(location.search).get('tag'); } catch (e) {}
    if (v) { return v; }
    var raw = location.hash ? location.hash.replace(/^#/, '') : '';
    if (!raw) { return null; }
    try { return decodeURIComponent(raw.replace(/\+/g, ' ')); } catch (e) { return null; }
  }

  function groupOf(tag) {
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].getAttribute('data-tag') === tag) { return groups[i]; }
    }
    return null;
  }

  function apply(tag, scroll) {
    var g = tag ? groupOf(tag) : null;
    if (tag && !g) { tag = null; g = null; }
    // 无过滤条件时不铺开所有标签分组，只保留上方标签云
    groups.forEach(function (el) {
      el.hidden = !g || el !== g;
      el.classList.toggle('is-solo', !!g && el === g);
    });
    var items = cloud ? cloud.querySelectorAll('.tag-item') : [];
    [].slice.call(items).forEach(function (a) {
      a.classList.toggle('is-active', !!g && a.getAttribute('data-tag') === tag);
    });
    // 单标签页：上方标签云整块让位，只留返回按钮 + 文章列表
    if (nav) { nav.hidden = !!g; }
    if (bar) { bar.hidden = !g; }
    // 独立页撤掉卡片外框，文章直接铺排（见 .is-filtered 规则）
    document.body.classList.toggle('is-filtered', !!g);

    if (g) {
      if (nameEl) { nameEl.textContent = tag; }
      if (countEl) { countEl.textContent = '（' + g.querySelectorAll('li').length + ' 篇）'; }
      if (h1) { h1.textContent = baseH1 + '：' + tag; }
      setDocTitle(baseH1 + '：' + tag);
    } else {
      if (h1) { h1.textContent = baseH1; }
      setDocTitle(baseH1);
    }
    if (scroll) { window.scrollTo({ top: 0, behavior: 'smooth' }); }
  }

  function setUrl(tag) {
    var url = location.pathname + (tag ? '?tag=' + encodeURIComponent(tag) : '');
    if (history.replaceState) { history.replaceState(null, '', url); }
  }

  if (cloud) {
    cloud.addEventListener('click', function (e) {
      var a = e.target && e.target.closest ? e.target.closest('.tag-item') : null;
      if (!a) { return; }
      e.preventDefault();
      var tag = a.getAttribute('data-tag');
      apply(tag, true);
      setUrl(tag);
    });
  }

  if (back) {
    back.addEventListener('click', function () {
      apply(null, true);
      setUrl(null);
    });
  }

  window.addEventListener('popstate', function () { apply(currentTag(), false); });
  window.addEventListener('hashchange', function () {
    // 旧式 #tag 链接：归一化成 ?tag= 参数
    var tag = currentTag();
    if (tag) { setUrl(tag); }
    apply(tag, false);
  });

  // 标签云：搜索 + 低频标签折叠
  var search = document.getElementById('tagSearch');
  var countInfo = document.getElementById('tagCountInfo');
  var showAllBtn = document.getElementById('tagShowAll');
  var cloudItems = cloud ? [].slice.call(cloud.querySelectorAll('.tag-item')) : [];
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
    showAllBtn.addEventListener('click', function () {
      showAll = !showAll;
      renderCloud();
    });
  }
  renderCloud();

  apply(currentTag(), false);
})();
</script>
