---
layout: page
title: 标签
---
{% assign tag_max = 0 %}
{% assign tag_min = 9999 %}
{% for tag in site.tags %}
  {% assign sz = tag[1] | size %}
  {% if sz > tag_max %}{% assign tag_max = sz %}{% endif %}
  {% if sz < tag_min %}{% assign tag_min = sz %}{% endif %}
{% endfor %}
{% assign colors = "#3b82f6,#6366f1,#8b5cf6,#a855f7,#ec4899,#e11d48,#ea580c,#d97706,#16a34a,#0d9488,#0891b2,#1d4ed8" | split: "," %}

  <!-- 浏览态：全站标签一览（进入单标签页时整体隐藏） -->
  <div id="tag-nav">
    <p class="page-desc">共 {{ site.tags | size }} 个标签、{{ site.posts | size }} 篇文章。<b>点击任意标签</b>，进入只列该标签文章的独立页面，地址栏带上 <code>?tag=</code> 参数，复制即可分享。</p>

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
        <a class="tag-item" href="/tags/?tag={{ tag[0] | url_encode }}" data-tag="{{ tag[0] }}" style="font-size: {{ font_pct }}%; color: {{ color }};" title="{{ tag_size }} 篇文章">{{ tag[0] }}</a>
      {% endfor %}
    </div>
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
    // 无过滤条件时 g 为 null，此时全部展开；有 g 时只有一个分组可见（独立页模式）
    groups.forEach(function (el) {
      el.hidden = !!g && (el !== g);
      el.classList.toggle('is-solo', !!g && el === g);
    });
    var items = cloud ? cloud.querySelectorAll('.tag-item') : [];
    [].slice.call(items).forEach(function (a) {
      a.classList.toggle('is-active', !!g && a.getAttribute('data-tag') === tag);
    });
    // 单标签页：上方标签云整块让位，只留返回按钮 + 文章列表
    if (nav) { nav.hidden = !!g; }
    if (bar) { bar.hidden = !g; }

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

  apply(currentTag(), false);
})();
</script>
