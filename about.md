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
  <img class="about-avatar" src="/images/touxiang.jpg" alt="{{ site.author }}">
  <div class="about-intro">
    <h2 class="about-name">{{ site.author }}<span class="about-handle">GitHub @lrypcy</span></h2>
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

<h2 class="about-section-title">联系我</h2>
<p class="about-note">
  文章有误、想讨论技术、或者单纯打个招呼，欢迎在任意文章底部留言（GitHub 账号登录即可评论），也可以到 <a href="https://github.com/lrypcy" target="_blank" rel="noopener">GitHub</a> 提 Issue。
</p>
