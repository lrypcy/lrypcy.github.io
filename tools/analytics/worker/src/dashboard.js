/**
 * 看板页渲染。全部在服务端拼成 HTML 字符串，不引入任何前端框架 / CDN 资源，
 * 这样在内网或被墙的环境里也能正常打开。
 */

import { percentOf } from './lib.js';
import { countryLabel, geoLabel, provinceLabel } from './names.js';

const esc = (s) =>
  String(s == null ? '' : s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
  );

const fmt = (n) => Number(n || 0).toLocaleString('en-US');

function shortLabel(key) {
  if (/^\d{4}-\d{2}-\d{2}$/.test(key)) return key.slice(5);
  if (/^\d{4}-\d{2}$/.test(key)) return key.slice(2);
  return key;
}

/** 用 CSS 高度做的柱状图：无依赖、可响应式、能容纳几百根柱子。 */
function bars(items, opts = {}) {
  const { height = 150, labelEvery = 1 } = opts;
  if (!items.length) return '<p class="empty">暂无数据</p>';
  const max = Math.max(...items.map((i) => Number(i.v) || 0), 1);
  const cells = items
    .map((item, idx) => {
      const value = Number(item.v) || 0;
      const pct = value > 0 ? Math.max((value / max) * 100, 1.5) : 0;
      const label = idx % labelEvery === 0 ? esc(shortLabel(item.k)) : '';
      return (
        `<div class="bar-cell" title="${esc(item.k)}：${fmt(value)} 次">` +
        `<div class="bar-wrap"><div class="bar" style="height:${pct.toFixed(2)}%"></div></div>` +
        `<div class="bar-label">${label}</div>` +
        `</div>`
      );
    })
    .join('');
  return `<div class="chart" style="--chart-h:${height}px">${cells}</div>`;
}

/** 横向条形列表，用于国家 / 地区排行。 */
function rankList(rows, titleOf, total) {
  if (!rows.length) return '<p class="empty">暂无数据</p>';
  const max = Math.max(...rows.map((r) => Number(r.v) || 0), 1);
  return `<ul class="rank">${rows
    .map((row) => {
      const value = Number(row.v) || 0;
      const width = Math.max((value / max) * 100, 1).toFixed(2);
      return (
        `<li class="rank-item">` +
        `<span class="rank-name" title="${esc(titleOf(row))}">${esc(titleOf(row))}</span>` +
        `<span class="rank-bar-track"><span class="rank-bar" style="width:${width}%"></span></span>` +
        `<span class="rank-value">${fmt(value)}<em>${percentOf(value, total).toFixed(1)}%</em></span>` +
        `</li>`
      );
    })
    .join('')}</ul>`;
}

function statCard(label, value, extra = '') {
  return (
    `<div class="stat"><div class="stat-label">${esc(label)}</div>` +
    `<div class="stat-value">${fmt(value)}</div>` +
    (extra ? `<div class="stat-extra">${esc(extra)}</div>` : '') +
    `</div>`
  );
}

function rangeTabs(token, range) {
  const tabs = [
    ['all', '全部时间'],
    ['year', '本年度'],
    ['month', '本月'],
    ['day', '今天'],
  ];
  return `<nav class="tabs">${tabs
    .map(
      ([key, text]) =>
        `<a class="${key === range ? 'active' : ''}" href="?token=${encodeURIComponent(token)}&range=${key}">${text}</a>`
    )
    .join('')}</nav>`;
}

export function renderDashboard({ token, range, day, stats, yearly, monthly, recent, geos }) {
  const monthSlice = monthly.slice(-24);
  const monthLabelEvery = Math.ceil(monthSlice.length / 12) || 1;
  const recentLabelEvery = Math.ceil(recent.length / 10) || 1;
  const yearLabelEvery = Math.ceil(yearly.length / 14) || 1;

  // 按范围取到的地理明细：先聚合成国家层，再保留明细行
  const byCountry = new Map();
  for (const row of geos) {
    const key = String(row.country || 'XX').toUpperCase();
    byCountry.set(key, (byCountry.get(key) || 0) + (Number(row.v) || 0));
  }
  const countryRows = [...byCountry.entries()]
    .map(([country, v]) => ({ country, v, title: countryLabel(country) }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 15);

  const rangeTotal = geos.reduce((sum, r) => sum + (Number(r.v) || 0), 0);
  const detailRows = geos.slice(0, 80).map(
    (row) =>
      `<tr><td>${esc(geoLabel(row))}</td><td class="num">${fmt(row.v)}</td><td class="num">${percentOf(
        row.v,
        rangeTotal
      ).toFixed(1)}%</td></tr>`
  );

  const rangeText = { all: '全部时间', year: '本年度', month: '本月', day: '今天' }[range] || range;

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>访问统计 · 一身骄傲</title>
<style>
  :root{
    --primary:#3b82f6; --bg:#f6f9fc; --card:#fff; --text:#1a2332; --muted:#6b7b8d;
    --border:#e6edf4; --track:#eef4fb;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:15px/1.7 "Noto Sans SC","PingFang SC",-apple-system,"Helvetica Neue",Arial,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:28px 20px 64px}
  header.top{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;
    margin-bottom:6px}
  h1{margin:0;font-size:22px;letter-spacing:.5px}
  .meta{color:var(--muted);font-size:13px}
  .tabs{display:flex;gap:6px;margin:18px 0 20px;flex-wrap:wrap}
  .tabs a{padding:6px 14px;border-radius:999px;background:var(--card);border:1px solid var(--border);
    color:var(--muted);text-decoration:none;font-size:13.5px;transition:.2s}
  .tabs a:hover{border-color:var(--primary);color:var(--primary)}
  .tabs a.active{background:var(--primary);border-color:var(--primary);color:#fff}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:22px}
  .stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:15px 17px;
    box-shadow:0 1px 3px rgba(30,40,60,.04)}
  .stat-label{color:var(--muted);font-size:13px}
  .stat-value{font-size:26px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
  .stat-extra{color:var(--muted);font-size:12.5px;margin-top:2px}
  section.card{background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 3px rgba(30,40,60,.04)}
  section.card > h2{margin:0 0 14px;font-size:15.5px;font-weight:600;display:flex;
    justify-content:space-between;align-items:baseline;gap:10px}
  section.card > h2 small{color:var(--muted);font-weight:400;font-size:12.5px}
  .chart{display:flex;align-items:flex-end;gap:2px;height:var(--chart-h,150px)}
  .bar-cell{flex:1 1 0;min-width:0;display:flex;flex-direction:column;justify-content:flex-end;height:100%}
  .bar-wrap{height:100%;display:flex;align-items:flex-end}
  .bar{width:100%;background:linear-gradient(180deg,#6ea6f7,#3b82f6);border-radius:3px 3px 0 0;
    min-height:1px;transition:.2s}
  .bar-cell:hover .bar{background:linear-gradient(180deg,#3b82f6,#2563eb)}
  .bar-label{font-size:10.5px;color:var(--muted);height:16px;line-height:16px;text-align:center;
    white-space:nowrap;overflow:hidden}
  .rank{list-style:none;margin:0;padding:0}
  .rank-item{display:flex;align-items:center;gap:10px;padding:5px 0}
  .rank-name{flex:0 0 200px;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .rank-bar-track{flex:1 1 auto;height:9px;background:var(--track);border-radius:999px;overflow:hidden}
  .rank-bar{display:block;height:100%;background:linear-gradient(90deg,#6ea6f7,#3b82f6);border-radius:999px}
  .rank-value{flex:0 0 108px;text-align:right;font-variant-numeric:tabular-nums;font-size:13.5px}
  .rank-value em{font-style:normal;color:var(--muted);font-size:12px;margin-left:7px}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th,td{padding:7px 8px;border-bottom:1px solid var(--border);text-align:left}
  th{color:var(--muted);font-weight:500;font-size:12.5px}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
  .empty{color:var(--muted);font-size:13.5px;margin:8px 0}
  details summary{cursor:pointer;color:var(--primary);font-size:13.5px;outline:none}
  footer.note{color:var(--muted);font-size:12.5px;line-height:1.9;margin-top:26px}
  footer.note code{background:#eef4fb;padding:1px 5px;border-radius:4px;font-size:12px}
  @media (max-width:600px){
    .rank-name{flex:0 0 110px}
    .rank-value{flex:0 0 76px;font-size:12.5px}
    .stat-value{font-size:22px}
  }
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <h1>访问统计</h1>
    <div class="meta">数据时区 UTC+8 · 统计日 ${esc(day)} · 范围「${esc(rangeText)}」</div>
  </header>
  ${rangeTabs(token, range)}

  <div class="stats">
    ${statCard('总访问次数', stats.total)}
    ${statCard('本年度', stats.year, `含今天在内的 ${esc(day.slice(0, 4))} 年`)}
    ${statCard('本月', stats.month, day.slice(0, 7))}
    ${statCard('今天', stats.today, `独立访客 ${fmt(stats.todayUv)}`)}
    ${statCard('本月独立访客', stats.monthUv, '按设备去重')}
  </div>

  <section class="card">
    <h2>年度访问次数 <small>共 ${yearly.length} 年</small></h2>
    ${bars(yearly, { height: 130, labelEvery: yearLabelEvery })}
  </section>

  <section class="card">
    <h2>月度访问次数 <small>最近 ${monthSlice.length} 个月</small></h2>
    ${bars(monthSlice, { height: 150, labelEvery: monthLabelEvery })}
  </section>

  <section class="card">
    <h2>最近 30 天 <small>按天</small></h2>
    ${bars(recent, { height: 130, labelEvery: recentLabelEvery })}
  </section>

  <section class="card">
    <h2>访问来源地区 <small>范围「${esc(rangeText)}」，共 ${fmt(rangeTotal)} 次</small></h2>
    ${rankList(countryRows, (row) => row.title, rangeTotal)}
  </section>

  <section class="card">
    <h2>地区明细 <small>省 / 城市级，最多 80 行</small></h2>
    <table>
      <thead><tr><th>地区</th><th class="num">访问次数</th><th class="num">占比</th></tr></thead>
      <tbody>${detailRows.join('') || '<tr><td colspan="3" class="empty">暂无数据</td></tr>'}</tbody>
    </table>
  </section>

  <section class="card">
    <h2>口径说明</h2>
    <footer class="note">
      <p><b>访问次数</b>：每次成功的页面上报记 1 次（PV）。同一访客 <code>DEDUPE_SECONDS</code> 秒内的重复上报会被丢弃；已知爬虫、监控与预览流量在服务端按 UA 直接过滤，空 UA 也丢弃。</p>
      <p><b>独立访客</b>：标识为 <code>SHA-256(盐 | 月份 | IP | UA)</code> 的截断值，<b>盐按月轮换</b>。因此「本月独立访客」准确，跨月的「年度独立访客」会把同一个人重复计入 —— 这是不长期跟踪个人的必然代价，本页因此不展示年度独立访客。</p>
      <p><b>地区</b>：来自 Cloudflare 对客户端 IP 的解析（<code>request.cf</code>），国家准确度高，中国的省级行政区基本可用，城市级在中国大陆质量一般、可能为空。<code>未知地区</code> 表示 Cloudflare 无法定位该 IP。</p>
      <p><b>存储</b>：不保存 IP、User-Agent、Cookie 与 localStorage，只保存「日期 × 地区」的计数行，以及当天的访客哈希。</p>
    </footer>
  </section>
</div>
</body>
</html>`;
}

export function renderUnauthorized() {
  return `<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow"><title>需要口令</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#f6f9fc;color:#1a2332;font:15px/1.7 "PingFang SC",-apple-system,Arial,sans-serif}
.box{background:#fff;border:1px solid #e6edf4;border-radius:12px;padding:26px 30px;max-width:420px;
box-shadow:0 1px 3px rgba(30,40,60,.05)}h1{margin:0 0 8px;font-size:18px}p{margin:6px 0;color:#6b7b8d;font-size:13.5px}
code{background:#eef4fb;padding:1px 5px;border-radius:4px}</style></head>
<body><div class="box"><h1>需要访问口令</h1>
<p>请在地址后加上 <code>?token=你的口令</code>。口令是部署时用 <code>wrangler secret put DASH_TOKEN</code> 设置的值。</p>
<p>建议把带口令的链接存成书签，本页不会被搜索引擎索引。</p></div></body></html>`;
}
