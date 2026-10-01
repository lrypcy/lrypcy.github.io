---
layout: page
title: 关于
permalink: /about/
---

{% assign graph = site.data.knowledge_graph %}
{% assign first_post = site.posts | last %}
{% assign now_ts = site.time | date: '%s' | plus: 0 %}
{% assign first_ts = first_post.date | date: '%s' | plus: 0 %}
{% assign write_days = now_ts | minus: first_ts | divided_by: 86400 | at_least: 1 %}

<div class="about-hero">
  <img class="about-avatar" src="/images/touxiang.jpg" alt="博主头像">
  <div class="about-intro">
    <h2 class="about-name"><span class="about-handle">GitHub @lrypcy</span></h2>
    <p class="about-tagline">{{ site.subtitle }}</p>
    <p class="about-bio">{{ site.about }}</p>
    <div class="about-links">
      <a href="https://github.com/lrypcy" target="_blank" rel="noopener">GitHub</a>
      <a href="mailto:p_c_yuan@whu.edu.cn">邮箱</a>
      <a href="/archives/">全部文章</a>
      <a href="/#series-quant">量化系列</a>
    </div>
  </div>
</div>

<div class="about-stats">
  <div class="about-stat"><b>{{ site.posts | size }}</b><span>篇文章</span></div>
  <div class="about-stat"><b>{{ graph.series | size }}</b><span>个系列</span></div>
  <div class="about-stat"><b>{{ site.categories | size }}</b><span>个分类</span></div>
  <div class="about-stat"><b>{{ site.tags | size }}</b><span>个标签</span></div>
  <div class="about-stat"><b>{{ write_days }}</b><span>天持续写作</span></div>
  <div class="about-stat"><b>{{ first_post.date | date: "%Y-%m" }}</b><span>开写于</span></div>
</div>

<h2 class="about-section-title">我在写什么</h2>
<p class="about-desc">所有文章按「分类 → 系列」两层组织，下面是当前的研究版图，随文章发布自动更新。</p>
<div class="about-cats">
  {% for c in graph.categories %}
  <a class="about-cat" href="/categories/#{{ c.name | url_encode }}" style="--cat-color: {{ c.color }}">
    <span class="about-cat-head">
      <span class="about-cat-icon">{{ c.icon }}</span>
      <span class="about-cat-name">{{ c.name }}</span>
      <span class="about-cat-count">{{ site.categories[c.name].size | default: 0 }} 篇</span>
    </span>
    <span class="about-cat-desc">{{ c.desc }}</span>
  </a>
  {% endfor %}
</div>

<h2 class="about-section-title">怎么读这个站</h2>
<ul class="about-ol">
  <li><b>首页知识图谱</b>：{{ graph.series | size }} 个系列、{{ site.posts | size }} 篇文章的依赖关系图，按「硬件与系统底座 → 核心算法 → 前沿专题」三层排布，节点可拖动，<a href="/">点这里去玩</a>。</li>
  <li><b>学习路线</b>：首页底部有四条由浅入深的推荐路线（AI Infra 工程师主线 / 大模型算法与后训练 / 具身智能与世界模型 / 生成模型与扩散）。</li>
  <li><b>系列导航</b>：每篇长文开头有「系列导航」引用块，结尾有上一篇 / 下一篇，顺着读不用回首页找。</li>
  <li><b>搜索</b>：右上角 ⌘K 全站全文搜索。</li>
</ul>

<h2 class="about-section-title">写作原则</h2>
<ul class="about-ol">
  <li><b>从原理到工程逐层拆解</b>：不止讲 What，更要讲 Why 和 How——公式逐条推导，代码动手实跑，实验数据直接回填进文章。</li>
  <li><b>每个算法都写两遍</b>：一份 from-scratch 手撸实现，一份对应官方库（PTQ / QAT 工具链）的实现，对照着才能真正理解。</li>
  <li><b>论断要可核验</b>：重要结论尽量附 arXiv / DOI / 官方文档链接，没把握的数据明确标注「待验证」。</li>
  <li><b>成体系而非碎片化</b>：宁可一个系列写透一个主题，不写零散的随笔。</li>
</ul>

{% assign ana = site.analytics %}
{% if ana and ana.enabled and ana.endpoint and ana.endpoint != "" %}

<h2 class="about-section-title">访问统计</h2>
<p class="about-desc">
  本站不接第三方统计：GA4 在国内基本加载不出来，Umami 免费版只保留 6 个月数据，GoatCounter 拿不到城市级别。
  所以这套统计跑在自己搭的 Cloudflare Worker + D1 上——只记录「日期 × 地区」的访问次数，不保存 IP、User-Agent 和 Cookie。
  下面是实时看板，也可以<a href="{{ ana.endpoint }}/stats" target="_blank" rel="noopener">单独打开</a>。
</p>
<div class="about-stats-embed">
<iframe id="pcy-stats-frame" src="{{ ana.endpoint }}/stats?embed=1" title="本站访问统计看板" loading="lazy" referrerpolicy="no-referrer" height="1500"></iframe>
</div>
<script>
/* 看板高度随「近 30 天柱数 / 地区明细行数」变化，写死会出现大片空白或内部滚动条。
   子页在 load / resize / ResizeObserver 时把自身高度 postMessage 过来，这里照着调 iframe。 */
(function () {
  var FRAME_ID = 'pcy-stats-frame';
  var MIN = 600, MAX = 6000;
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || data.type !== 'pcy-analytics:height') return;
    var frame = document.getElementById(FRAME_ID);
    if (!frame) return;
    var height = Number(data.height);
    if (!isFinite(height) || height < MIN || height > MAX) return;
    frame.style.height = height + 'px';
    frame.removeAttribute('height');
  });
})();
</script>

{% endif %}

<h2 class="about-section-title">联系我</h2>
<p class="about-note">
  文章有误、想讨论技术、或者单纯打个招呼，欢迎在任意文章底部留言（GitHub 账号登录即可评论），也可以到 <a href="https://github.com/lrypcy" target="_blank" rel="noopener">GitHub</a> 提 Issue。
</p>

<div class="about-wechat">
  <div class="about-wechat-qr">
    <img src="/images/wechat-aiinfra-group.png" alt="AI infra 交流群二维码" loading="lazy">
  </div>
  <div class="about-wechat-info">
    <h3>🧑‍💻 AI infra 交流群</h3>
    <p>量化、编译器、算子、分布式训练、RL——聊得来的都在群里，欢迎来吹水与切磋。</p>
    <p class="about-wechat-tip">⚠️ 微信群二维码 <b>7 天有效</b>。如果二维码已过期，可以加我微信<b>备注「lrypcy」</b>，看到后我会手动拉你进群。</p>
  </div>
</div>
