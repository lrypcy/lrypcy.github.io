/**
 * 看板页渲染。除「访问地图」的腾讯地图 SDK 外，全部在服务端拼成 HTML 字符串，
 * 不引入任何前端框架 / 构建产物 —— 离线或被墙时，去掉地图卡片仍能完整打开。
 */

import { percentOf } from './lib.js';
import { countryLabel, geoLabel } from './names.js';
import { buildProvinceStats } from './province-map.js';

/* ---------------------------------------------------------------------------
 * 访问地图：腾讯地图 GL JS（合规白名单内的地图来源）。
 *
 * 【为什么要自备 key】
 * 腾讯 / 高德 / 百度 / 天地图都需要 key；WorkBuddy 的免 key 代理（_TMapSecurityConfig）
 * 只在它的预览环境里把 `__WB_HTTP_PORT__` / `__WB_TMAP_SECRET__` 换成真值，
 * 部署到 Cloudflare Worker 后这两个占位符原样保留、代理地址不存在，所以线上必须用
 * 博主自己在腾讯位置服务申请的 key。申请与配置步骤见 tools/analytics/README.md。
 *
 * key 未配置（为空，或仍是源码里的占位符）时，「访问地图」卡片整块不渲染 —— 不给访客
 * 看空白地图，也不把开发说明泄露到页面上。
 * ------------------------------------------------------------------------- */

/** 占位符：真实 key 由部署方经 Worker 变量 / 密钥 `TMAP_KEY` 注入，源码里不放可用 key。 */
const TMAP_KEY_PLACEHOLDER =
  'Please apply for your own key at the 腾讯位置服务 Open Platform and replace this placeholder';

/** 没配 key、或 key 还是占位符，都算「未配置」。 */
const tmapKeyUnset = (key) => !key || /^Please apply/.test(key);

/** 全国视野：中心点约国土几何中心，zoom 4 基本铺满中国。 */
const MAP_CENTER = [35.4, 104.5];
const MAP_ZOOM = 4;

/**
 * 地图引导脚本：注入数据 + key，加载 SDK，打气泡与标签。
 *
 * 气泡用内联的 `data:image/svg+xml` 作为 MarkerStyle.src（官方 demo 的外部图片路径不可引用），
 * 每个档位一个样式；数字标签走 MultiLabel，位置与气泡同心。
 */
function mapScript(points, mapKey) {
  const payload = JSON.stringify(points).replace(/</g, '\\u003c');
  const key = JSON.stringify(mapKey);
  return `<script type="application/json" id="pcy-map-data">${payload}</script>
<script>
(function () {
  var el = document.getElementById('pcy-map-canvas');
  if (!el) return;
  var data = [];
  try { data = JSON.parse(document.getElementById('pcy-map-data').textContent || '[]'); } catch (e) {}
  var COLORS = { 1: '#93c5fd', 2: '#60a5fa', 3: '#3b82f6', 4: '#1d4ed8' };
  var SIZES = { 1: 15, 2: 21, 3: 29, 4: 39 };
  function bubble(color) {
    var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">' +
      '<circle cx="20" cy="20" r="17" fill="' + color + '" fill-opacity="0.82" ' +
      'stroke="#ffffff" stroke-width="2.5"/></svg>';
    return 'data:image/svg+xml,' + encodeURIComponent(svg);
  }
  function boot() {
    if (!window.TMap) return;
    var map = new TMap.Map(el, {
      center: new TMap.LatLng(${MAP_CENTER[0]}, ${MAP_CENTER[1]}),
      zoom: ${MAP_ZOOM}
    });
    var styles = {};
    for (var l = 1; l <= 4; l++) {
      var s = SIZES[l];
      styles['b' + l] = new TMap.MarkerStyle({
        width: s, height: s, anchor: { x: s / 2, y: s / 2 }, src: bubble(COLORS[l])
      });
    }
    new TMap.MultiMarker({
      id: 'pcy-markers',
      map: map,
      styles: styles,
      geometries: data.map(function (p, i) {
        return {
          id: 'm' + i,
          styleId: 'b' + (p.lv || 1),
          position: new TMap.LatLng(p.lat, p.lng),
          properties: { title: p.n + ' · ' + p.v.toLocaleString('en-US') + ' 次访问' }
        };
      })
    });
    new TMap.MultiLabel({
      id: 'pcy-labels',
      map: map,
      styles: {
        c: new TMap.LabelStyle({
          color: '#0f172a', size: 12, offset: { x: 0, y: -2 },
          alignment: 'center', verticalAlignment: 'middle'
        })
      },
      geometries: data.map(function (p, i) {
        return {
          id: 'l' + i,
          styleId: 'c',
          position: new TMap.LatLng(p.lat, p.lng),
          content: p.n + ' ' + p.v.toLocaleString('en-US')
        };
      })
    });
  }
  var s = document.createElement('script');
  s.src = 'https://map.qq.com/api/gljs?v=1.exp&key=' + encodeURIComponent(${key});
  s.onload = boot;
  s.onerror = function () { el.innerHTML = '<div class="mapbox-tip">访问地图加载失败</div>'; };
  document.head.appendChild(s);
})();
</script>`;
}

const esc = (s) =>
  String(s == null ? '' : s).replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]
  );

const fmt = (n) => Number(n || 0).toLocaleString('en-US');

/**
 * 内嵌模式的高度上报：把自身文档高度 postMessage 给父页面，由父页调整 iframe 高度。
 *
 * 不写死像素高度 —— 内容高度会随「近 30 天柱状图」和「地区明细行数」变化，
 * 写死要么留一大片空白，要么在 iframe 内部出现滚动条（移动端尤其难看）。
 * 消息体只有一个数字，不含任何统计数据，用 '*' 作为 targetOrigin 是安全的。
 */
const EMBED_HEIGHT_SCRIPT = `<script>
(function () {
  var last = 0;
  function report() {
    var h = Math.ceil(document.documentElement.getBoundingClientRect().height);
    if (!h || h === last) return;
    last = h;
    try { parent.postMessage({ type: 'pcy-analytics:height', height: h }, '*'); } catch (e) {}
  }
  window.addEventListener('load', report);
  window.addEventListener('resize', report);
  if (window.ResizeObserver) new ResizeObserver(report).observe(document.body);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(report).catch(function () {});
})();
</script>`;

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

function rangeTabs({ token, range, embed }) {
  const tabs = [
    ['all', '全部时间'],
    ['year', '本年度'],
    ['month', '本月'],
    ['day', '今天'],
  ];
  // 公开内嵌模式下不能把口令带进链接，否则会随页面源码一起漏出去。
  const hrefFor = (key) => {
    const query = [];
    if (token) query.push(`token=${encodeURIComponent(token)}`);
    if (embed) query.push('embed=1');
    query.push(`range=${key}`);
    return `?${query.join('&')}`;
  };
  return `<nav class="tabs">${tabs
    .map(([key, text]) => `<a class="${key === range ? 'active' : ''}" href="${hrefFor(key)}">${text}</a>`)
    .join('')}</nav>`;
}

/**
 * GitHub 贡献图式的活动日历。
 *
 * 用 CSS grid 的 `grid-auto-flow: column` 铺格子：列 = 周、行 = 周内第几天，
 * 这样只需给每列 7 个格子按顺序输出，浏览器自动按列排布，不必手算绝对坐标。
 * 空档（还没到的日子）给透明格子，与「当天 0 次访问」的浅色格子区分开。
 */
function heatmap(cal) {
  if (!cal || !cal.weeks.length) return '<p class="empty">暂无数据</p>';

  const labelByCol = new Map(cal.monthLabels.map((m) => [m.col, m.label]));
  const monthCells = cal.weeks
    .map((_, col) => `<span class="hm-month">${esc(labelByCol.get(col) || '')}</span>`)
    .join('');

  const cells = cal.weeks
    .map((week) =>
      week
        .map((cell) => {
          if (!cell) return '<span class="hm-cell hm-void"></span>';
          const text = cell.pv > 0 ? `${cell.day} · ${fmt(cell.pv)} 次访问` : `${cell.day} · 无访问`;
          return `<span class="hm-cell l${cell.level}" title="${esc(text)}"></span>`;
        })
        .join('')
    )
    .join('');

  const legend = Array.from({ length: 5 }, (_, l) => `<span class="hm-cell l${l}"></span>`).join('');
  const summary =
    cal.activeDays > 0
      ? `近 ${cal.weeks.length} 周有 ${fmt(cal.activeDays)} 天有访问，单日最高 ${fmt(cal.max)} 次`
      : `近 ${cal.weeks.length} 周暂无访问`;

  return `<div class="hm">
  <div class="hm-scroll">
    <div class="hm-inner">
      <div class="hm-months">${monthCells}</div>
      <div class="hm-body">
        <div class="hm-days"><span></span><span>一</span><span></span><span>三</span><span></span><span>五</span><span></span></div>
        <div class="hm-grid">${cells}</div>
      </div>
    </div>
  </div>
  <div class="hm-legend"><span>少</span>${legend}<span>多</span><span class="hm-summary">${esc(summary)}</span></div>
</div>`;
}

/**
 * 省级访问地图卡片。key 未配置（仍是占位符）时返回空串，调用方据此整块不渲染。
 */
function provinceMap(stats, mapKey) {
  if (tmapKeyUnset(mapKey)) return '';
  if (!stats || !stats.items.length) return '<p class="empty">暂无数据</p>';

  const points = stats.items.map((p) => ({ n: p.name, v: p.v, lat: p.lat, lng: p.lng, lv: p.level }));

  const parts = [`${stats.activeCount} 个省级行政区有访问，共 ${fmt(stats.mapped)} 次`];
  if (stats.unmapped > 0) parts.push(`另有 ${fmt(stats.unmapped)} 次只定位到国家、未定位到省份`);

  // 图例色块必须与 mapScript 里 COLORS 的 4 个档位同序同色，改动时两边一起改。
  const legend = ['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8']
    .map((c) => `<i style="background:${c}"></i>`)
    .join('');

  return `<div id="pcy-map-canvas" class="mapbox-canvas"></div>
${mapScript(points, mapKey)}
<div class="mp-legend"><span>访问量 少</span><b class="mp-scale">${legend}</b><span>多</span></div>
<p class="mp-summary">${esc(parts.join('；'))}</p>
<p class="mp-note">气泡圆心为省级行政区中心点（<code>GCJ-02</code>），圆越大、颜色越深表示访问越多；港澳台与大陆同图。</p>`;
}

export function renderDashboard({
  token = '',
  embed = false,
  range,
  day,
  stats,
  yearly,
  monthly,
  recent,
  geos,
  calendar,
  tmapKey = '',
}) {
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

  // 访问地图直接吃 geos，不需要额外的 SQL —— 落库维度里已经有 country / province。
  const provinceStats = buildProvinceStats(geos);
  const mapTotal = provinceStats.mapped + provinceStats.unmapped;
  const mapHtml = provinceMap(provinceStats, tmapKey || TMAP_KEY_PLACEHOLDER);

  const detailRows = geos.slice(0, 80).map(
    (row) =>
      `<tr><td>${esc(geoLabel(row))}</td><td class="num">${fmt(row.v)}</td><td class="num">${percentOf(
        row.v,
        rangeTotal
      ).toFixed(1)}%</td></tr>`
  );

  const rangeText = { all: '全部时间', year: '本年度', month: '本月', day: '今天' }[range] || range;
  const weekCount = calendar ? calendar.weeks.length : 0;

  // 内嵌到关于页时口径说明会把 iframe 撑得很高，压缩成一行脚注；
  // 单独打开看板时保留完整说明 —— 公开页上「统计了什么」本身就该讲清楚。
  const notes = embed
    ? `<p class="footnote">热力图每格一天，颜色按当日访问次数分档，空格子表示那天还没到；
       访问地图底图为腾讯地图，气泡位置为省级行政区中心点，颜色深浅按访问量分档。
       只记录「日期 × 地区」的计数，不保存 IP、User-Agent 与 Cookie。</p>`
    : `<section class="card">
    <h2>口径说明</h2>
    <footer class="note">
      <p><b>访问次数</b>：每次成功的页面上报记 1 次（PV）。同一访客 <code>DEDUPE_SECONDS</code> 秒内的重复上报会被丢弃；已知爬虫、监控与预览流量在服务端按 UA 直接过滤，空 UA 也丢弃。</p>
      <p><b>独立访客</b>：标识为 <code>SHA-256(盐 | 月份 | IP | UA)</code> 的截断值，<b>盐按月轮换</b>。因此「本月独立访客」准确，跨月的「年度独立访客」会把同一个人重复计入 —— 这是不长期跟踪个人的必然代价，本页因此不展示年度独立访客。</p>
      <p><b>地区</b>：来自 Cloudflare 对客户端 IP 的解析（<code>request.cf</code>），国家准确度高，中国的省级行政区基本可用，城市级在中国大陆质量一般、可能为空。<code>未知地区</code> 表示 Cloudflare 无法定位该 IP。</p>
      <p><b>热力图</b>：每格一天，颜色深浅按当日访问次数占「近 ${weekCount} 周内单日最高值」的比例分 4 档；空格子表示那一天还没到，不是零访问。</p>
      <p><b>访问地图</b>：底图与疆界由腾讯地图提供，按国家标准绘制（含南海诸岛与九段线）；气泡圆心是省级行政区中心点（<code>GCJ-02</code>），大小与颜色按访问量分档。落点依据是 Cloudflare 返回的 <code>regionCode</code>；若只返回英文省名（如 <code>Guangdong</code>）或没有省级信息，这部分流量只汇总进「未定位到省份」，不会落到图上。港澳台与大陆同图。地图由浏览器直接向 <code>map.qq.com</code> 请求底图，会带上本站域名作为 Referer —— 这是整个看板唯一的外部依赖。</p>
      <p><b>存储</b>：不保存 IP、User-Agent、Cookie 与 localStorage，只保存「日期 × 地区」的计数行，以及当天的访客哈希。</p>
    </footer>
  </section>`;

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

  /* ---- 活动热力图（GitHub 贡献图式） ---- */
  .hm{--hm-cell:11px;--hm-gap:3px;--hm-label-w:22px;margin-top:4px}
  .hm-scroll{overflow-x:auto;padding-bottom:6px}
  .hm-inner{display:inline-block;min-width:100%}
  .hm-months{display:grid;grid-auto-flow:column;grid-auto-columns:var(--hm-cell);
    gap:var(--hm-gap);margin:0 0 5px var(--hm-label-w);
    font-size:9.5px;line-height:13px;color:var(--muted)}
  .hm-month{white-space:nowrap;overflow:visible}
  .hm-body{display:flex;gap:6px;align-items:flex-start}
  .hm-days{display:grid;grid-template-rows:repeat(7,var(--hm-cell));gap:var(--hm-gap);
    width:calc(var(--hm-label-w) - 6px);font-size:9.5px;line-height:var(--hm-cell);
    color:var(--muted);text-align:right}
  .hm-grid{display:grid;grid-auto-flow:column;grid-template-rows:repeat(7,var(--hm-cell));
    grid-auto-columns:var(--hm-cell);gap:var(--hm-gap)}
  .hm-cell{display:block;width:var(--hm-cell);height:var(--hm-cell);border-radius:2.5px;
    background:#eef2f7;outline:1px solid rgba(27,49,80,.05);outline-offset:-1px}
  .hm-cell.l1{background:#c6e0fb}
  .hm-cell.l2{background:#8cc0f7}
  .hm-cell.l3{background:#4f9bef}
  .hm-cell.l4{background:#1f6fe0}
  .hm-cell.hm-void{background:transparent;outline:none}
  .hm-legend{display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-top:12px;
    font-size:12px;color:var(--muted)}
  .hm-legend .hm-cell{display:inline-block}
  .hm-legend > span{margin-right:2px}
  .hm-summary{margin-left:8px;padding-left:10px;border-left:1px solid var(--border)}

  /* ---- 访问地图（腾讯地图 GL JS；容器必须有固定高度，否则地图不渲染） ---- */
  .mapbox-canvas{height:420px;border-radius:10px;overflow:hidden;background:#eef2f7}
  .mapbox-tip{height:100%;display:flex;align-items:center;justify-content:center;
    color:var(--muted);font-size:13px}
  .mp-summary{color:var(--muted);font-size:12.5px;margin:10px 0 0}
  /* 图例的 4 个色块与地图气泡的 4 个档位一一对应（色值见 mapScript 里的 COLORS） */
  .mp-legend{display:flex;align-items:center;gap:8px;margin-top:10px;font-size:12px;color:var(--muted)}
  .mp-legend .mp-scale{display:inline-flex;gap:3px}
  .mp-legend .mp-scale i{display:block;width:15px;height:12px;border-radius:3px}
  .mp-note{color:var(--muted);font-size:12px;line-height:1.75;margin:8px 0 0}

  @media (max-width:600px){
    .rank-name{flex:0 0 110px}
    .rank-value{flex:0 0 76px;font-size:12.5px}
    .stat-value{font-size:22px}
    .hm{--hm-cell:9px;--hm-gap:2.5px}
    .mapbox-canvas{height:300px}
  }

  /* ---- 内嵌模式（关于页 iframe） ---- */
  body.embed{background:transparent}
  body.embed .wrap{max-width:none;padding:0}
  body.embed h1{font-size:17px}
  /* 关于页正文列比看板窄，格子缩小一档，免得在 iframe 里还要横向滚动 */
  body.embed .hm{--hm-cell:10px;--hm-gap:2.5px;--hm-label-w:20px}
  /* 关于页正文列更窄，地图矮一档，免得在 iframe 里占掉整屏 */
  body.embed .mapbox-canvas{height:320px}
  @media (max-width:600px){
    body.embed .hm{--hm-cell:9px;--hm-gap:2px;--hm-label-w:18px}
    body.embed .mapbox-canvas{height:260px}
  }
  body.embed .stats{grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:10px;margin-bottom:14px}
  body.embed .stat{padding:11px 13px;border-radius:10px}
  body.embed .stat-value{font-size:21px}
  body.embed section.card{padding:14px 16px;margin-bottom:12px;border-radius:10px}
  body.embed section.card > h2{font-size:14.5px;margin-bottom:10px}
  .footnote{color:var(--muted);font-size:12.5px;line-height:1.8;margin:14px 2px 2px}
</style>
</head>
<body${embed ? ' class="embed"' : ''}>
<div class="wrap">
  <header class="top">
    ${embed ? '' : '<h1>访问统计</h1>'}
    <div class="meta">数据时区 UTC+8 · 统计日 ${esc(day)} · 范围「${esc(rangeText)}」</div>
  </header>
  ${rangeTabs({ token, range, embed })}

  <div class="stats">
    ${statCard('总访问次数', stats.total)}
    ${statCard('本年度', stats.year, `含今天在内的 ${esc(day.slice(0, 4))} 年`)}
    ${statCard('本月', stats.month, day.slice(0, 7))}
    ${statCard('今天', stats.today, `独立访客 ${fmt(stats.todayUv)}`)}
    ${statCard('本月独立访客', stats.monthUv, '按设备去重')}
  </div>

  <section class="card">
    <h2>活动热力图 <small>最近 ${weekCount} 周，每格一天</small></h2>
    ${heatmap(calendar)}
  </section>

  ${mapHtml ? `<section class="card">
    <h2>访问地图 <small>省级 · 范围「${esc(rangeText)}」共 ${fmt(mapTotal)} 次</small></h2>
    ${mapHtml}
  </section>` : ''}

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
    <h2>国家 / 地区排行 <small>范围「${esc(rangeText)}」，共 ${fmt(rangeTotal)} 次</small></h2>
    ${rankList(countryRows, (row) => row.title, rangeTotal)}
    <p class="mp-note">这里按国家 / 地区汇总，中国大陆各省的分布见上方地图。</p>
  </section>

  <section class="card">
    <h2>地区明细 <small>省 / 城市级，最多 80 行</small></h2>
    <table>
      <thead><tr><th>地区</th><th class="num">访问次数</th><th class="num">占比</th></tr></thead>
      <tbody>${detailRows.join('') || '<tr><td colspan="3" class="empty">暂无数据</td></tr>'}</tbody>
    </table>
  </section>

  ${notes}
</div>
${embed ? EMBED_HEIGHT_SCRIPT : ''}
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
