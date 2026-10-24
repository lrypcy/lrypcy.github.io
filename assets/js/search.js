(function () {
  var overlay = document.getElementById('searchOverlay');
  var input = document.getElementById('searchInput');
  var results = document.getElementById('searchResults');
  var trigger = document.querySelector('.search-trigger');
  var posts = null;
  var activeIndex = -1;
  var typeFilter = 'all';
  var lastQuery = '';

  function open() {
    overlay.classList.add('is-open');
    input.value = '';
    typeFilter = 'all';
    results.innerHTML = '<div class="search-empty">输入关键词开始搜索…</div>';
    activeIndex = -1;
    setTimeout(function () { input.focus(); }, 80);
  }

  function close() {
    overlay.classList.remove('is-open');
    input.value = '';
    results.innerHTML = '';
    activeIndex = -1;
  }

  trigger.addEventListener('click', open);
  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); open(); }
    if (e.key === 'Escape' && overlay.classList.contains('is-open')) { e.preventDefault(); close(); }
  });
  overlay.addEventListener('click', function (e) { if (e.target === overlay) close(); });

  function loadIndex(cb) {
    if (posts) return cb();
    fetch('/search-index.json')
      .then(function (r) { return r.json(); })
      .then(function (data) { posts = data; cb(); })
      .catch(function () {
        results.innerHTML = '<div class="search-empty">搜索索引加载失败</div>';
      });
  }

  function doSearch(query) {
    if (!query || !posts) return [];
    var q = query.toLowerCase();
    var scored = [];
    for (var i = 0; i < posts.length; i++) {
      var p = posts[i];
      var title = (p.title || '').toLowerCase();
      var content = (p.content || '').toLowerCase();
      var tags = (p.tags || []).join(' ').toLowerCase();
      var cats = (p.categories || []).join(' ').toLowerCase();
      var score = 0;
      if (title.indexOf(q) !== -1) score += 10;
      if (tags.indexOf(q) !== -1) score += 5;
      if (cats.indexOf(q) !== -1) score += 3;
      if (content.indexOf(q) !== -1) score += 1;
      if (score > 0) scored.push({ post: p, score: score });
    }
    scored.sort(function (a, b) { return b.score - a.score; });
    return scored.map(function (s) { return s.post; });
  }

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlight(text, query, maxLen) {
    if (!query || !text) return text || '';
    var safe = escapeHtml(text);
    var qLower = query.toLowerCase();
    var idx = safe.toLowerCase().indexOf(qLower);
    if (idx === -1) {
      return safe.length > maxLen ? safe.slice(0, maxLen) + '…' : safe;
    }
    var start = Math.max(0, idx - 40);
    var end = Math.min(safe.length, idx + query.length + 60);
    var snippet = (start > 0 ? '…' : '') + safe.slice(start, end) + (end < safe.length ? '…' : '');
    var re = new RegExp('(' + query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
    return snippet.replace(re, '<mark>$1</mark>');
  }

  function renderResults(matches, query) {
    var filtered = matches.filter(function (p) {
      if (typeFilter === 'all') { return true; }
      return p.type === (typeFilter === 'news' ? '资讯' : '文章');
    });
    var html = typeChipsHtml(matches);
    if (filtered.length === 0) {
      var scope = typeFilter === 'news' ? '资讯' : (typeFilter === 'post' ? '文章' : '内容');
      html += '<div class="search-empty">该类型下未找到匹配「' + escapeHtml(query) + '」的' + scope + '</div>';
      results.innerHTML = html;
      activeIndex = -1;
      return;
    }
    for (var i = 0; i < filtered.length; i++) {
      var p = filtered[i];
      var snippet = highlight(p.content, query, 120);
      var typeBadge = p.type ? '<span class="search-result-type' + (p.type === '资讯' ? ' is-news' : '') + '">' + p.type + '</span>' : '';
      html += '<a class="search-result-item" href="' + p.url + '" data-idx="' + i + '">' +
        '<div class="search-result-title">' + highlight(p.title, query, 80) + '</div>' +
        '<div class="search-result-snippet">' + snippet + '</div>' +
        '<div class="search-result-meta">' +
          typeBadge +
          '<span class="search-result-date">' + p.date + '</span>' +
          (p.categories && p.categories.length ? ' · ' + p.categories.join(', ') : '') +
        '</div>' +
      '</a>';
    }
    results.innerHTML = html;
    activeIndex = -1;
  }

  function typeChipsHtml(matches) {
    var counts = { all: matches.length, post: 0, news: 0 };
    for (var i = 0; i < matches.length; i++) {
      if (matches[i].type === '资讯') { counts.news++; } else { counts.post++; }
    }
    function chip(key, label, n) {
      return '<button type="button" class="search-type-chip' + (typeFilter === key ? ' is-active' : '') + '" data-type="' + key + '">' + label + ' ' + n + '</button>';
    }
    return '<div class="search-type-bar">' + chip('all', '全部', counts.all) + chip('post', '文章', counts.post) + chip('news', '资讯', counts.news) + '</div>';
  }

  input.addEventListener('input', function () {
    var q = input.value.trim();
    lastQuery = q;
    if (!q) {
      results.innerHTML = '<div class="search-empty">输入关键词开始搜索…</div>';
      return;
    }
    loadIndex(function () {
      renderResults(doSearch(q), q);
    });
  });

  results.addEventListener('click', function (e) {
    var chip = e.target && e.target.closest ? e.target.closest('.search-type-chip') : null;
    if (!chip) { return; }
    e.preventDefault();
    typeFilter = chip.getAttribute('data-type');
    if (lastQuery) { renderResults(doSearch(lastQuery), lastQuery); }
  });

  input.addEventListener('keydown', function (e) {
    var items = results.querySelectorAll('.search-result-item');
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, items.length - 1);
      updateActive(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, -1);
      updateActive(items);
    } else if (e.key === 'Enter' && activeIndex >= 0 && items[activeIndex]) {
      e.preventDefault();
      window.location.href = items[activeIndex].getAttribute('href');
    }
  });

  function updateActive(items) {
    for (var i = 0; i < items.length; i++) {
      items[i].classList.toggle('is-active', i === activeIndex);
    }
    if (activeIndex >= 0 && items[activeIndex]) {
      items[activeIndex].scrollIntoView({ block: 'nearest' });
    }
  }
})();
