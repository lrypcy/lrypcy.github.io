/**
 * 生成「访问地图」设计预览页 —— 不用申请腾讯 key 就能看地图长什么样。
 *
 * 做法：直接调用 Worker 真实的渲染函数 `renderDashboard()`（所以版式、气泡、
 * 标签、图例与线上逐像素同源），只把两处改成预览模式：
 *   1. 在 <head> 注入 WorkBuddy 免 key 代理配置 `_TMapSecurityConfig`；
 *   2. 把 SDK 地址上的 `&key=<key>` 摘掉（代理模式带 key 反而会白屏）。
 *
 * 于是这个页面**只在 WorkBuddy 的预览里能看到底图**；用系统浏览器打开会是灰底，
 * 那是预期的 —— 它只是设计预览，不是线上方案。线上必须自备 key，见 README §3.3。
 *
 * 用法：
 *   node tools/analytics/make-map-preview.mjs [输出路径]
 *   # 默认输出 tools/analytics/map-preview.html（已 gitignore）
 *   # 生成后交给 present_files 打开即可看到真实底图
 *
 * 页内数据全部是**示例值**，与 D1 无关；真实数据只有 /stats 或 /dash 才有。
 */

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { renderDashboard } from './worker/src/dashboard.js';
import { buildCalendar, HEATMAP_WEEKS } from './worker/src/lib.js';

const OUT = process.argv[2] || fileURLToPath(new URL('./map-preview.html', import.meta.url));

/* ---------- 示例数据：只为把地图填满，好判断版式与色阶 ---------- */
const GEO = [
  ['CN', 'BJ', 'Beijing', 7730],
  ['CN', 'SH', 'Shanghai', 5902],
  ['CN', 'GD', 'Shenzhen', 5576],
  ['CN', 'ZJ', 'Hangzhou', 3410],
  ['CN', 'JS', 'Nanjing', 2988],
  ['CN', 'SC', 'Chengdu', 2140],
  ['CN', 'HB', 'Wuhan', 1876],
  ['CN', 'SD', 'Jinan', 1520],
  ['CN', 'HN', 'Changsha', 1204],
  ['CN', 'FJ', 'Xiamen', 980],
  ['CN', 'SN', "Xi'an", 742],
  ['CN', 'CQ', 'Chongqing', 655],
  ['CN', 'LN', 'Shenyang', 431],
  ['CN', 'YN', 'Kunming', 388],
  ['CN', 'XJ', 'Urumqi', 212],
  ['CN', 'TW', 'Taipei', 176],
  ['HK', '', 'Hong Kong', 420],
  ['SG', '', 'Singapore', 96],   // 不在图内，用来验证海外流量确实不进省级地图
  ['US', 'CA', 'San Jose', 74],  // 同上
];

const geos = GEO.map(([country, province, city, v]) => ({ country, province, city, v }));
const total = geos.reduce((sum, g) => sum + g.v, 0);
const day = '2026-10-01';

const recent = Array.from({ length: 30 }, (_, i) => {
  const d = new Date(Date.UTC(2026, 8, 2 + i));
  return { k: d.toISOString().slice(0, 10), v: Math.round(140 + 70 * Math.sin(i / 3.2) + (i % 7 === 0 ? 150 : 0)) };
});
const monthly = Array.from({ length: 14 }, (_, i) => {
  const m = 2025 * 12 + 7 + i;
  return { k: `${Math.floor(m / 12)}-${String((m % 12) + 1).padStart(2, '0')}`, v: 400 + i * 460 };
});

const html = renderDashboard({
  token: '',
  embed: false,
  range: 'all',
  day,
  stats: { total, today: 268, month: 6120, year: total, todayUv: 231, monthUv: 4180 },
  yearly: [
    { k: '2025', v: Math.round(total * 0.42) },
    { k: '2026', v: total },
  ],
  monthly,
  recent,
  geos,
  calendar: buildCalendar(
    [
      { k: '2026-09-18', v: 240 }, { k: '2026-09-19', v: 610 }, { k: '2026-09-20', v: 190 },
      { k: '2026-09-24', v: 430 }, { k: '2026-09-27', v: 780 }, { k: '2026-09-29', v: 320 },
      { k: '2026-10-01', v: 900 },
    ],
    { endDay: day, weeks: HEATMAP_WEEKS }
  ),
  tmapKey: 'SAMPLE-PREVIEW-KEY',
});

/* ---------- 1. 摘掉 SDK 上的 key（占位符原样保留，由 WorkBuddy 运行时替换） ---------- */
const PROXY =
  "<script>window._TMapSecurityConfig={serviceHost:'http://127.0.0.1:__WB_HTTP_PORT__/_TMapService/_wbt/__WB_TMAP_SECRET__'};</script>";

const withoutKey = html.replace(
  /'https:\/\/map\.qq\.com\/api\/gljs\?v=1\.exp&key='\s*\+\s*encodeURIComponent\([^)]*\)/,
  "'https://map.qq.com/api/gljs?v=1.exp'"
);
if (withoutKey === html) {
  throw new Error('没能摘掉 SDK 的 key —— 生成中止，避免把带 key 的页面当成预览');
}

/* ---------- 2. 代理配置必须在 SDK 加载之前执行，所以塞进 <head> 尾部 ---------- */
const withProxy = withoutKey.replace('</head>', `${PROXY}\n</head>`);
if (withProxy === withoutKey) throw new Error('没找到 </head>，无法注入代理配置');

/* ---------- 3. 顶部放一条醒目横幅，避免把示例数据当成真实数据 ---------- */
const BANNER =
  '<div style="padding:10px 14px;margin-bottom:14px;border-radius:10px;background:#fff7ed;' +
  'border:1px solid #fdba74;color:#9a3412;font-size:13px;line-height:1.7">' +
  '<b>设计预览</b>：页内数据是<b>示例值</b>，只为填满图表。底图走 WorkBuddy 内置免 key 代理，' +
  '所以只在 WorkBuddy 预览里显示；线上要自备腾讯位置服务的 key（<code>TMAP_KEY</code>，见 README §3.3）。' +
  '</div>';

const final = withProxy.replace(/(<body[^>]*>)/, `$1\n${BANNER}`);
writeFileSync(OUT, final);

console.log(`已生成 ${OUT}（${final.length} bytes）`);
console.log('  含地图容器:', /pcy-map-canvas/.test(final));
console.log('  含图例    :', /class="mp-legend"/.test(final));
console.log('  SDK 已去 key:', /gljs\?v=1\.exp'/.test(final));
console.log('  已注入代理 :', final.includes('_TMapSecurityConfig'));
